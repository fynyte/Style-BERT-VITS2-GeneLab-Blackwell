import argparse
import datetime
import gc
import os
import platform
from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from huggingface_hub import HfApi
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler

# logging.getLogger("numba").setLevel(logging.WARNING)
import default_style
from config import get_config
from data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import WavLMLoss, discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.attentions import MultiHeadAttention
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.modules import ConvFlow
from style_bert_vits2.models.transforms import piecewise_rational_quadratic_transform
from style_bert_vits2.models.models_jp_extra import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_num_threads(1)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # Not available if torch version is lower than 2.0

config = get_config()
global_step = 0

api = HfApi()


class _CompactRelativeToAbsolute(torch.autograd.Function):
    """Scatter only the learned relative offsets, without a (2*T-1) padding."""

    @staticmethod
    def forward(ctx, x):
        length, width = x.shape[-2:]
        window = width // 2
        ctx.length, ctx.window = length, window
        out = x.new_zeros(*x.shape[:-1], length)
        for offset in range(-window, window + 1):
            start, end = max(-offset, 0), min(length, length - offset)
            out.diagonal(offset, dim1=-2, dim2=-1).copy_(
                x[..., start:end, offset + window]
            )
        return out

    @staticmethod
    def backward(ctx, grad):
        length, window = ctx.length, ctx.window
        out = grad.new_zeros(*grad.shape[:-1], 2 * window + 1)
        for offset in range(-window, window + 1):
            start, end = max(-offset, 0), min(length, length - offset)
            out[..., start:end, offset + window].copy_(
                grad.diagonal(offset, dim1=-2, dim2=-1)
            )
        return out

class _CompactAbsoluteToRelative(torch.autograd.Function):
    """Gather the local diagonals used by relative values, including gradients."""

    @staticmethod
    def forward(ctx, x, window):
        ctx.length, ctx.window = x.shape[-1], window
        out = x.new_zeros(*x.shape[:-1], 2 * window + 1)
        for offset in range(-window, window + 1):
            start, end = max(-offset, 0), min(ctx.length, ctx.length - offset)
            out[..., start:end, offset + window].copy_(
                x.diagonal(offset, dim1=-2, dim2=-1)
            )
        return out

    @staticmethod
    def backward(ctx, grad):
        length, window = ctx.length, ctx.window
        out = grad.new_zeros(*grad.shape[:-1], length)
        for offset in range(-window, window + 1):
            start, end = max(-offset, 0), min(length, length - offset)
            out.diagonal(offset, dim1=-2, dim2=-1).copy_(
                grad[..., start:end, offset + window]
            )
        return out, None

def _compact_relative_embeddings(module, embeddings, length):
    window = min(module.window_size, length - 1)
    return embeddings[:, module.window_size - window:module.window_size + window + 1]

def _compact_relative_to_absolute(module, x):
    return _CompactRelativeToAbsolute.apply(x)

def _compact_absolute_to_relative(module, x):
    return _CompactAbsoluteToRelative.apply(x, min(module.window_size, x.shape[-1] - 1))

def _release_attention_reference(module, inputs, output):
    # The training graph retains what backward needs. No training code reads this
    # diagnostic attribute; keeping it also holds the previous batch's matrices.
    module.attn = None

def _install_compact_relative_attention(net):
    count = 0
    for module in net.modules():
        if isinstance(module, MultiHeadAttention) and module.window_size is not None:
            module._get_relative_embeddings = MethodType(_compact_relative_embeddings, module)
            module._relative_position_to_absolute_position = MethodType(
                _compact_relative_to_absolute, module
            )
            module._absolute_position_to_relative_position = MethodType(
                _compact_absolute_to_relative, module
            )
            module.register_forward_hook(_release_attention_reference)
            count += 1
    return count

def can_freeze_discriminator_for_generator(
    discriminator, *, use_spectral_norm, duration_discriminator=None,
    wavlm_discriminator=None, world_size=None,
):
    """Return false for configurations outside the numerically validated scope."""
    if use_spectral_norm or duration_discriminator is not None or wavlm_discriminator is not None:
        return False
    if world_size is not None and int(world_size) != 1:
        return False
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        return False
    if isinstance(discriminator, nn.DataParallel):
        return False
    if isinstance(discriminator, nn.parallel.DistributedDataParallel):
        if torch.distributed.get_world_size(discriminator.process_group) != 1:
            return False
        discriminator = discriminator.module
    # Check the actual legacy spectral-norm buffers as well as the config flag.
    return not any("weight_u" in module._buffers for module in discriminator.modules())

@contextmanager
def generator_discriminator_pass(
    discriminator, *, enabled=True, use_spectral_norm=False,
    duration_discriminator=None, wavlm_discriminator=None, world_size=None,
):
    """Yield the D call for G and restore every original flag on all exits.

    The caller must include the G loss backward in this context. At one DDP
    rank, the underlying module keeps the derivative with respect to generated
    audio without asking DDP to prepare an unused D parameter reduction.
    Unsupported configurations retain the normal discriminator call.
    """
    allowed = enabled and can_freeze_discriminator_for_generator(
        discriminator, use_spectral_norm=use_spectral_norm,
        duration_discriminator=duration_discriminator,
        wavlm_discriminator=wavlm_discriminator, world_size=world_size,
    )
    if not allowed:
        yield discriminator
        return
    module = discriminator.module if isinstance(discriminator, nn.parallel.DistributedDataParallel) else discriminator
    parameters = tuple(module.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def run():
    # Command line configuration is not recommended unless necessary, use config.yml
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=config.train_ms_config.config_path,
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
        default=config.dataset_path,
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Root directory of model assets needed for inference.",
        default=config.assets_root,
    )
    parser.add_argument(
        "--skip_default_style",
        action="store_true",
        help="Skip saving default style config and mean vector.",
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not show the progress bar while training.",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Speed up training by disabling logging and evaluation.",
    )
    parser.add_argument(
        "--repo_id",
        help="Huggingface model repo id to backup the model.",
        default=None,
    )
    parser.add_argument(
        "--not_use_custom_batch_sampler",
        help="Don't use custom batch sampler for training, which was used in the version < 2.5",
        action="store_true",
    )
    parser.add_argument(
        "--disable_bf16_optimizations",
        action="store_true",
        help="Use the original relative-position attention arithmetic for comparison.",
    )
    parser.add_argument(
        "--disable_algorithm_optimizations",
        action="store_true",
        help="Use the original mel, spline, prior expansion and WavLM algorithms for comparison.",
    )
    parser.add_argument(
        "--compile_spline",
        action="store_true",
        help="Compile the shared dense training spline (Linux/CUDA with Triton; first step includes compilation).",
    )
    args = parser.parse_args()

    # Set log file
    model_dir = os.path.join(args.model, config.train_ms_config.model_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(os.path.join(args.model, f"train_{timestamp}.log"))

    # Parsing environment variables
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            logger.info(f"Loading configuration from config {env_value!s}")
            os.environ[env_name] = str(env_value)
    logger.info(
        "Loading environment variables \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = dist.get_world_size()

    hps = HyperParameters.load_from_json(args.config)
    # This is needed because we have to pass values to `train_and_evaluate()
    hps.model_dir = model_dir
    hps.speedup = args.speedup
    hps.algorithm_optimizations = not args.disable_algorithm_optimizations
    if args.compile_spline and not hps.algorithm_optimizations:
        parser.error("--compile_spline requires algorithm optimizations")
    hps.repo_id = args.repo_id

    # 比较路径是否相同
    if os.path.realpath(args.config) != os.path.realpath(
        config.train_ms_config.config_path
    ):
        with open(args.config, encoding="utf-8") as f:
            data = f.read()
        os.makedirs(os.path.dirname(config.train_ms_config.config_path), exist_ok=True)
        with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
            f.write(data)

    """
    Path constants are a bit complicated...
    TODO: Refactor or rename these?
    (Both `config.yml` and `config.json` are used, which is confusing I think.)

    args.model: For saving all info needed for training.
        default: `Data/{model_name}`.
    hps.model_dir := model_dir: For saving checkpoints (for resuming training).
        default: `Data/{model_name}/models`.
        (Use `hps` since we have to pass `model_dir` to `train_and_evaluate()`.

    args.assets_root: The root directory of model assets needed for inference.
        default: config.assets_root == `model_assets`.

    config.out_dir: The directory for model assets of this model (for inference).
        default: `model_assets/{model_name}`.
    """

    if args.repo_id is not None:
        # First try to upload config.json to check if the repo exists
        try:
            api.upload_file(
                path_or_fileobj=args.config,
                path_in_repo=f"Data/{config.model_name}/config.json",
                repo_id=hps.repo_id,
            )
        except Exception as e:
            logger.error(e)
            logger.error(
                f"Failed to upload files to the repo {hps.repo_id}. Please check if the repo exists and you have logged in using `huggingface-cli login`."
            )
            raise e
        # Upload Data dir for resuming training
        api.upload_folder(
            repo_id=hps.repo_id,
            folder_path=config.dataset_path,
            path_in_repo=f"Data/{config.model_name}",
            delete_patterns="*.pth",  # Only keep the latest checkpoint
            ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
            run_as_future=True,
        )
    os.makedirs(config.out_dir, exist_ok=True)

    if not args.skip_default_style:
        default_style.save_styles_by_dirs(
            os.path.join(args.model, "wavs"),
            config.out_dir,
            config_path=args.config,
            config_output_path=os.path.join(config.out_dir, "config.json"),
        )

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(local_rank)

    global global_step
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        # logger = utils.get_logger(hps.model_dir)
        # logger.info(hps)
        utils.check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate(use_jp_extra=True)
    if not args.not_use_custom_batch_sampler:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            # batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
    else:
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            # shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
        logger.info("Using DistributedLengthGroupedSampler for training.")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None
    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).cuda(local_rank)
    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing (JP) bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        logger.info("Freezing style encoder !!!")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_decoder", False):
        logger.info("Freezing decoder !!!")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    if hps.train.bf16_run and not args.disable_bf16_optimizations:
        relative_count = _install_compact_relative_attention(net_g)
        logger.info(f"BF16 compact relative attention enabled: {relative_count} layers")

    if hps.algorithm_optimizations:
        net_g.use_gather_prior = True
        spline_transform = (
            torch.compile(
                piecewise_rational_quadratic_transform, fullgraph=True, dynamic=True
            )
            if args.compile_spline
            else None
        )
        for module in net_g.sdp.modules():
            if isinstance(module, ConvFlow):
                module.use_dense_spline = True
                if spline_transform is not None:
                    module.spline_transform = spline_transform
        logger.info(
            "Algorithm optimizations enabled: segment mel, dense spline, gather prior, shared WavLM"
        )
        if args.compile_spline:
            logger.info("Shared spline compilation enabled; first step includes warmup")

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None
    net_g = DDP(
        net_g,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    net_d = DDP(
        net_d,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            # bucket_cap_mb=512,
        )
    if net_wd is not None:
        net_wd = DDP(
            net_wd,
            device_ids=[local_rank],
            #  bucket_cap_mb=512
        )

    net_d._sbv2_freeze_for_generator = (
        hps.train.bf16_run
        and not args.disable_bf16_optimizations
        and can_freeze_discriminator_for_generator(
            net_d, use_spectral_norm=hps.model.use_spectral_norm,
            duration_discriminator=net_dur_disc, wavlm_discriminator=net_wd,
            world_size=n_gpus,
        )
    )
    if net_d._sbv2_freeze_for_generator:
        logger.info("Generator update: skip unused discriminator parameter gradients")

    if utils.is_resuming(model_dir):
        if net_dur_disc is not None:
            try:
                _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    utils.checkpoints.get_latest_checkpoint_path(
                        model_dir, "DUR_*.pth"
                    ),
                    net_dur_disc,
                    optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer,
                )
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
            except:
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
                print("Initialize dur_disc")
        if net_wd is not None:
            try:
                _, optim_wd, wd_resume_lr, epoch_str = (
                    utils.checkpoints.load_checkpoint(
                        utils.checkpoints.get_latest_checkpoint_path(
                            model_dir, "WD_*.pth"
                        ),
                        net_wd,
                        optim_wd,
                        skip_optimizer=hps.train.skip_optimizer,
                    )
                )
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
            except:
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
                logger.info("Initialize wavlm")

        try:
            _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr

            epoch_str = max(epoch_str, 1)
            # global_step = (epoch_str - 1) * len(train_loader)
            global_step = int(
                utils.get_steps(
                    utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
                )
            )
            logger.info(
                f"******************Found the model. Current epoch is {epoch_str}, gloabl step is {global_step}*********************"
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
            epoch_str = 1
            global_step = 0
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "G_0.safetensors"), net_g
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "DUR_0.safetensors"), net_dur_disc
                )
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "WD_0.safetensors"), net_wd
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch):
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
    else:
        scheduler_dur_disc = None
    if net_wd is not None:
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None
    if hps.train.fp16_run and hps.train.bf16_run:
        logger.warning(
            "Both fp16_run and bf16_run are set to True in config.json; bf16_run takes precedence."
        )
    if hps.train.bf16_run:
        logger.info("Mixed precision training: bf16")
    elif hps.train.fp16_run:
        logger.info("Mixed precision training: fp16 (GradScaler enabled)")
    else:
        logger.info("Mixed precision training: disabled (fp32)")
    # GradScaler は fp16 の勾配アンダーフロー対策としてのみ必要。
    # bf16 は fp32 相当の指数レンジを持つため scaler は不要 (enabled=False で問題ない)。
    scaler = GradScaler(enabled=hps.train.fp16_run and not hps.train.bf16_run)
    logger.info("Start training.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    pbar = None
    if not args.no_progress_bar:
        pbar = tqdm(
            total=global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        )
    initial_step = global_step

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                pbar,
                initial_step,
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, None],
                None,
                None,
                pbar,
                initial_step,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()
        if net_wd is not None:
            scheduler_wd.step()
        if epoch == hps.train.epochs:
            # Save the final models
            assert optim_g is not None
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"),
            )
            assert optim_d is not None
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"DUR_{global_step}.pth"),
                )
            if net_wd is not None:
                assert optim_wd is not None
                utils.checkpoints.save_checkpoint(
                    net_wd,
                    optim_wd,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"WD_{global_step}.pth"),
                )
            utils.safetensors.save_safetensors(
                net_g,
                epoch,
                os.path.join(
                    config.out_dir,
                    f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True,
            )
            if hps.repo_id is not None:
                future1 = api.upload_folder(
                    repo_id=hps.repo_id,
                    folder_path=config.dataset_path,
                    path_in_repo=f"Data/{config.model_name}",
                    delete_patterns="*.pth",  # Only keep the latest checkpoint
                    ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
                    run_as_future=True,
                )
                future2 = api.upload_folder(
                    repo_id=hps.repo_id,
                    folder_path=config.out_dir,
                    path_in_repo=f"model_assets/{config.model_name}",
                    run_as_future=True,
                )
                try:
                    future1.result()
                    future2.result()
                except Exception as e:
                    logger.error(e)

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    pbar: tqdm,
    initial_step: int,
):
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # 混合精度モードの決定: bf6_run を優先し、次に fp16_run、両方 False なら fp32 (AMP無効)。
    # T4 (Turing, sm_75) は bf16 の Tensor Core 演算に対応していないため、
    # T4環境では config.json で fp16_run: true を指定することを推奨する。
    if hps.train.bf16_run:
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif hps.train.fp16_run:
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        amp_dtype = torch.float32
        amp_enabled = False

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        style_vec,
    ) in enumerate(train_loader):
        # 勾配ノルムはログ用のみ。通常ステップでは計測しないことで、GPU 同期を
        # 発生させない。rank 0 以外はこの値を出力しないため計測不要。
        log_this_step = (
            rank == 0
            and global_step % hps.train.log_interval == 0
            and not hps.speedup
        )
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(
            local_rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(
            local_rank, non_blocking=True
        ), spec_lengths.cuda(local_rank, non_blocking=True)
        y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(
            local_rank, non_blocking=True
        )
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        style_vec = style_vec.cuda(local_rank, non_blocking=True)

        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),  # , logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
            )
            segment_frames = hps.train.segment_size // hps.data.hop_length
            mel_input = (
                commons.slice_segments(spec, ids_slice, segment_frames)
                if hps.algorithm_optimizations
                else spec
            )
            mel = spec_to_mel_torch(
                mel_input,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = (
                mel
                if hps.algorithm_optimizations
                else commons.slice_segments(mel, ids_slice, segment_frames)
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
                # The decoder's final tanh guarantees this range. Avoid two
                # host synchronizations checking an invariant of y_hat.
                check_range=not hps.algorithm_optimizations,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                # torch.nn.utils.clip_grad_norm_(
                # parameters=net_dur_disc.parameters(), max_norm=5
                # )
                scaler.step(optim_dur_disc)
            wavlm_real_embeddings = None
            if net_wd is not None:
                # logger.debug(f"y.shape: {y.shape}, y_hat.shape: {y_hat.shape}")
                # shape: (batch, 1, time)
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    if (
                        hps.algorithm_optimizations
                        and wl.can_share_embeddings()
                        and (net_dur_disc is not None or log_this_step)
                    ):
                        loss_slm, wavlm_real_embeddings = wl.discriminator(
                            y.detach().squeeze(1),
                            y_hat.detach().squeeze(1),
                            return_real_embeddings=True,
                        )
                    else:
                        loss_slm = wl.discriminator(
                            y.detach().squeeze(1), y_hat.detach().squeeze(1)
                        ).mean()

                optim_wd.zero_grad()
                scaler.scale(loss_slm).backward()
                scaler.unscale_(optim_wd)
                # torch.nn.utils.clip_grad_norm_(parameters=net_wd.parameters(), max_norm=200)
                if log_this_step:
                    grad_norm_wd = commons.clip_grad_value_(
                        net_wd.parameters(), None
                    )
                scaler.step(optim_wd)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        if amp_enabled:
            # fp16/bf16 いずれの混合精度時も、Discriminatorの勾配爆発対策として norm clipping を適用する。
            # 特にfp16はbf16よりダイナミックレンジが狭くNaN化しやすいため、この安全策の意味が大きい。
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        if log_this_step:
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with generator_discriminator_pass(
            net_d, enabled=net_d._sbv2_freeze_for_generator,
            use_spectral_norm=hps.model.use_spectral_norm,
            duration_discriminator=net_dur_disc, wavlm_discriminator=net_wd,
        ) as g_discriminator:
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = g_discriminator(y, y_hat)
                if net_dur_disc is not None:
                    _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
                if net_wd is not None:
                    # The existing objective includes these losses only when
                    # the duration discriminator is enabled. Otherwise they
                    # are diagnostics and are needed only on logging steps.
                    if hps.algorithm_optimizations:
                        if net_dur_disc is not None:
                            loss_lm, loss_lm_gen = wl.generator_losses(
                                y.detach().squeeze(1),
                                y_hat.squeeze(1),
                                wav_embeddings=wavlm_real_embeddings,
                            )
                        elif log_this_step:
                            with torch.no_grad():
                                loss_lm, loss_lm_gen = wl.generator_losses(
                                    y.detach().squeeze(1),
                                    y_hat.squeeze(1),
                                    wav_embeddings=wavlm_real_embeddings,
                                )
                    else:
                        loss_lm = wl(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                        loss_lm_gen = wl.generator(y_hat.squeeze(1))
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    # loss_commit = loss_commit * hps.train.c_commit

                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                    if net_dur_disc is not None:
                        loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                        if net_wd is not None:
                            loss_gen_all += loss_dur_gen + loss_lm + loss_lm_gen
                        else:
                            loss_gen_all += loss_dur_gen
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            if amp_enabled:
                # Discriminator側と同様、fp16/bf16 いずれの混合精度時も
                # Generatorの勾配爆発対策として norm clipping を適用する(fp32時は元の挙動通り適用しない)。
                torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
            if log_this_step:
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if log_this_step:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # logger.info(
                #     "Train Epoch: {} [{:.0f}%]".format(
                #         epoch, 100.0 * batch_idx / len(train_loader)
                #     )
                # )
                # logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                )

                if net_dur_disc is not None:
                    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

                    scalar_dict.update(
                        {
                            f"loss/dur_disc_g/{i}": v
                            for i, v in enumerate(losses_dur_disc_g)
                        }
                    )
                    scalar_dict.update(
                        {
                            f"loss/dur_disc_r/{i}": v
                            for i, v in enumerate(losses_dur_disc_r)
                        }
                    )

                    scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                    scalar_dict.update(
                        {f"loss/g/dur_gen_{i}": v for i, v in enumerate(losses_dur_gen)}
                    )

                if net_wd is not None:
                    scalar_dict.update(
                        {
                            "loss/wd/total": loss_slm,
                            "grad_norm_wd": grad_norm_wd,
                            "loss/g/lm": loss_lm,
                            "loss/g/lm_gen": loss_lm_gen,
                        }
                    )
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )

            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
            ):
                if not hps.speedup:
                    evaluate(hps, net_g, eval_loader, writer_eval)
                utils.checkpoints.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth"),
                )
                if net_dur_disc is not None:
                    utils.checkpoints.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"DUR_{global_step}.pth"),
                    )
                if net_wd is not None:
                    utils.checkpoints.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"WD_{global_step}.pth"),
                    )
                keep_ckpts = config.train_ms_config.keep_ckpts
                if keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )
                # Save safetensors (for inference) to `model_assets/{model_name}`
                utils.safetensors.save_safetensors(
                    net_g,
                    epoch,
                    os.path.join(
                        config.out_dir,
                        f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )
                if hps.repo_id is not None:
                    api.upload_folder(
                        repo_id=hps.repo_id,
                        folder_path=config.dataset_path,
                        path_in_repo=f"Data/{config.model_name}",
                        delete_patterns="*.pth",  # Only keep the latest checkpoint
                        ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
                        run_as_future=True,
                    )
                    api.upload_folder(
                        repo_id=hps.repo_id,
                        folder_path=config.out_dir,
                        path_in_repo=f"model_assets/{config.model_name}",
                        run_as_future=True,
                    )

        global_step += 1
        if pbar is not None:
            pbar.set_description(
                f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}"
            )
            pbar.update()

    gc.collect()
    torch.cuda.empty_cache()
    if pbar is None and rank == 0:
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print()
    logger.info("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # mel = spec_to_mel_torch(
                #     spec,
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # y_hat_mel = mel_spectrogram_torch(
                #     y_hat.squeeze(1).float(),
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.hop_length,
                #     hps.data.win_length,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # image_dict.update(
                #     {
                #         f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             y_hat_mel[0].cpu().numpy()
                #         )
                #     }
                # )
                # image_dict.update(
                #     {
                #         f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             mel[0].cpu().numpy()
                #         )
                #     }
                # )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    run()
