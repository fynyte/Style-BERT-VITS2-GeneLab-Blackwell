"""Behavior/gradient regressions, executed on Modal (not the local machine)."""
import math

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

from style_bert_vits2.models.models_jp_extra import DurationPredictor, SynthesizerTrn
from style_bert_vits2.models.prosody import ProsodyAdapter
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.prosody_features import extract_prosody, load_or_extract_prosody


def tiny_model(enabled=True):
    return SynthesizerTrn(
        len(SYMBOLS), 17, 4, inter_channels=16, hidden_channels=32,
        filter_channels=64, n_heads=2, n_layers=3, kernel_size=3,
        p_dropout=0.0, resblock="1", resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]], upsample_rates=[2, 2],
        upsample_initial_channel=32, upsample_kernel_sizes=[4, 4],
        n_speakers=1, gin_channels=16, n_flow_layer=1, n_layers_trans_flow=3,
        use_noise_scaled_mas=False, use_prosody_adapter=enabled,
    )


def inputs(device="cpu"):
    return (
        torch.randint(1, len(SYMBOLS), (2, 6), device=device),
        torch.tensor([6, 4], device=device),
        torch.zeros(2, dtype=torch.long, device=device),
        torch.zeros(2, 6, dtype=torch.long, device=device),
        torch.zeros(2, 6, dtype=torch.long, device=device),
        torch.randn(2, 1024, 6, device=device),
        torch.randn(2, 256, device=device),
    )


def test_f0_tracks_known_frequency_and_frame_centers():
    sr, hop, nfft = 44100, 512, 2048
    audio = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    features = extract_prosody(audio, sr, hop, nfft, len(audio) // hop)
    assert features.shape == (3, len(audio) // hop)
    voiced = features[2] > 0
    hz = np.exp(features[0, voiced] * 0.5 + math.log(180))
    assert voiced.mean() > 0.9
    assert abs(np.median(hz) - 220) < 4


def test_silence_is_finite_and_unvoiced():
    result = extract_prosody(np.zeros(44100), 44100, 512, 2048, 86)
    assert np.isfinite(result).all()
    assert (result[0] == 0).all() and (result[2] == 0).all()


def test_cache_invalidates_when_audio_changes(tmp_path):
    import soundfile as sf
    path = tmp_path / "a.wav"
    audio = np.zeros(44100)
    sf.write(path, audio, 44100)
    first = load_or_extract_prosody(path, audio, 44100, 512, 2048, 86)
    # Valid cache: must not re-extract even if caller provides another array.
    cached = load_or_extract_prosody(path, np.ones_like(audio), 44100, 512, 2048, 86)
    np.testing.assert_array_equal(first, cached)
    audio = 0.2 * np.sin(2 * np.pi * 250 * np.arange(88200) / 44100)
    sf.write(path, audio, 44100)
    changed = load_or_extract_prosody(path, audio, 44100, 512, 2048, 172)
    assert changed[2].mean() > 0.9


def test_pooling_does_not_average_f0_with_unvoiced_frames():
    frames = torch.tensor([[[2., 0., 4., 0.], [1., 1., 2., 2.], [1., 0., 1., 0.]]])
    path = torch.tensor([[[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]]])
    target, voiced = ProsodyAdapter.targets(frames, path, path.sum(2))
    torch.testing.assert_close(target[:, 0], torch.tensor([[2., 4.]]))
    torch.testing.assert_close(voiced, torch.full((1, 1, 2), 0.5))


def test_modes_are_selected_without_blending():
    adapter = ProsodyAdapter(8, 4, 2)
    means = torch.stack((torch.full((3, 5), -2.), torch.full((3, 5), 2.)))[None]
    prediction = (torch.tensor([[3., -3.]]), means, torch.zeros_like(means), torch.zeros(1, 1, 5))
    _, automatic = adapter.infer_features(prediction, torch.ones(1, 1, 5))
    _, explicit = adapter.infer_features(prediction, torch.ones(1, 1, 5), component=1)
    assert (automatic == -2).all() and (explicit == 2).all()
    with pytest.raises(ValueError):
        adapter.infer_features(prediction, torch.ones(1, 1, 5), component=2)


def test_duration_loss_reaches_new_style_projection_past_legacy_detach():
    adapter = ProsodyAdapter(8, 4)
    dp = DurationPredictor(8, 16, 3, 0.0)
    hidden = torch.randn(2, 8, 5, requires_grad=True)
    style = torch.randn(2, 256)
    output = dp(hidden, torch.ones(2, 1, 5), style_condition=adapter.condition_duration(style))
    output.square().mean().backward()
    assert hidden.grad is None  # preserve the old stability boundary
    assert adapter.duration_style.weight.grad.abs().sum() > 0


def test_distribution_loss_reaches_text_and_ignores_padding():
    torch.manual_seed(1)
    adapter = ProsodyAdapter(8, 4)
    hidden = torch.randn(2, 8, 5, requires_grad=True)
    mask = torch.ones(2, 1, 5)
    mask[1, :, 3:] = 0
    prediction = adapter.predict(hidden, mask, torch.randn(2, 256))
    target = torch.randn(2, 3, 5)
    voiced = torch.rand(2, 1, 5)
    loss, _, _ = adapter.training_loss_and_features(prediction, target, voiced, mask)
    target[1, :, 3:] = 10000
    changed, _, _ = adapter.training_loss_and_features(prediction, target, voiced, mask)
    torch.testing.assert_close(loss, changed)
    loss.backward()
    assert hidden.grad.abs().sum() > 0
    assert adapter.statistics.weight.grad.abs().sum() > 0


def test_legacy_checkpoint_initial_output_matches_exactly():
    torch.manual_seed(2)
    legacy = tiny_model(False).eval()
    enhanced = tiny_model(True).eval()
    result = enhanced.load_state_dict(legacy.state_dict(), strict=False)
    assert result.missing_keys and all(k.startswith("prosody_adapter.") for k in result.missing_keys)
    assert not result.unexpected_keys
    arguments = inputs()
    with torch.no_grad():
        torch.manual_seed(5)
        old = legacy.infer(*arguments, noise_scale=0., noise_scale_w=0.)
        torch.manual_seed(5)
        new = enhanced.infer(*arguments, noise_scale=0., noise_scale_w=0.)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_gpu_forward_backward_and_prior_supervision(dtype):
    from losses import kl_loss
    torch.manual_seed(9)
    model = tiny_model().cuda().train()
    phone, lengths, sid, tone, language, bert, style = inputs("cuda")
    spec = torch.rand(2, 17, 24, device="cuda")
    frame_target = torch.randn(2, 3, 24, device="cuda")
    frame_target[:, 2] = torch.rand(2, 24, device="cuda") > 0.3
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        output = model(phone, lengths, spec, torch.tensor([24, 20], device="cuda"),
                       sid, tone, language, bert, style, prosody_targets=frame_target)
        z, z_p, m_p, logs_p, m_q, logs_q = output[6]
        loss = (output[0].float().square().mean() + output[1].float().sum()
                + output[9] + kl_loss(z_p, logs_q, m_p, logs_p, output[5]))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Unused parameter under DDP: {name}"
        assert torch.isfinite(p.grad).all(), name
    assert model.prosody_adapter.prior_residual.weight.grad.abs().sum() > 0
    assert model.prosody_adapter.duration_style.weight.grad.abs().sum() > 0
    assert model.enc_p.style_proj.weight.grad.abs().sum() > 0


def test_collate_sorts_prosody_with_its_audio():
    from data_utils import TextAudioSpeakerCollate
    def row(frames, marker):
        return (torch.ones(3, dtype=torch.long), torch.zeros(17, frames),
                torch.zeros(1, frames * 4), torch.tensor([0]),
                torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.zeros(1024, 3), torch.zeros(256), torch.full((3, frames), marker))
    result = TextAudioSpeakerCollate(use_jp_extra=True)([row(4, 1.), row(7, 2.)])
    assert len(result) == 12
    assert (result[11][0] == 2).all()
    assert (result[11][1, :, :4] == 1).all()
    assert (result[11][1, :, 4:] == 0).all()


def test_old_optimizer_migrates_once_and_new_optimizer_resumes(tmp_path):
    from style_bert_vits2.models.utils.checkpoints import load_checkpoint, save_checkpoint
    old = tiny_model(False)
    path = tmp_path / "G_1.pth"
    save_checkpoint(old, torch.optim.AdamW(old.parameters()), 1e-4, 1, path)
    new = tiny_model(True)
    optimizer = torch.optim.AdamW(new.parameters())
    load_checkpoint(path, new, optimizer)
    torch.testing.assert_close(new.enc_p.emb.weight, old.enc_p.emb.weight)
    save_checkpoint(new, optimizer, 1e-4, 2, path)
    load_checkpoint(path, new, optimizer)


def test_modified_python_sources_compile():
    import py_compile
    from pathlib import Path
    for path in [Path("train_ms_jp_extra.py"), Path("convert_onnx.py"),
                 Path("data_utils.py"), Path("style_bert_vits2/tts_model.py"),
                 *Path("style_bert_vits2/models").rglob("*.py")]:
        py_compile.compile(str(path), doraise=True)


def test_wavlm_backpropagates_in_real_trainer_without_duration_discriminator(tmp_path):
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from style_bert_vits2.models.hyper_parameters import HyperParameters
    import train_ms_jp_extra as trainer

    class SmallDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 2, 3, padding=1)

        def forward(self, real, generated):
            r, g = self.conv(real), self.conv(generated)
            return [r], [g], [[r]], [[g]]

    callbacks = []
    wd = nn.Conv1d(1, 1, 1).cuda()

    class SmallWavLMLoss:
        def can_share_embeddings(self):
            return True

        def discriminator(self, real, generated, return_real_embeddings=False):
            loss = wd(real[:, None]).square().mean()
            return (loss, None) if return_real_embeddings else loss

        def generator_losses(self, real, generated, wav_embeddings=None):
            loss = generated.square().mean()
            loss.register_hook(lambda grad: callbacks.append(float(grad)))
            return loss, (wd(generated[:, None]) - 1).square().mean()

    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    try:
        model = DDP(tiny_model().cuda(), device_ids=[0])
        discriminator = DDP(SmallDiscriminator().cuda(), device_ids=[0])
        discriminator._sbv2_freeze_for_generator = False
        hps = HyperParameters()
        hps.model.use_prosody_adapter = True
        hps.speedup = True
        hps.train.segment_size = 16
        hps.data.filter_length = hps.data.win_length = 32
        hps.data.hop_length = 4
        hps.data.n_mel_channels = 8
        hps.data.sampling_rate = 16000
        x, lengths, sid, tone, lang, bert, style = inputs()
        target = torch.randn(2, 3, 24)
        target[:, 2] = 1
        batch = (x, lengths, torch.rand(2, 17, 24), torch.tensor([24, 20]),
                 torch.rand(2, 1, 96) * 0.2, torch.tensor([96, 80]), sid, tone, lang,
                 bert, style, target)
        optimizers = [torch.optim.AdamW(net.parameters(), lr=1e-4)
                      for net in (model, discriminator, wd)]
        trainer.global_step = 0
        trainer.train_and_evaluate(
            0, 0, 1, hps, (model, discriminator, None, wd, SmallWavLMLoss()),
            (optimizers[0], optimizers[1], None, optimizers[2]), (None,) * 4,
            torch.cuda.amp.GradScaler(enabled=False), ([batch], None),
            trainer.logger, None, None, 0,
        )
        assert callbacks == [1.0]
        assert trainer.global_step == 1
    finally:
        dist.destroy_process_group()


def test_enhanced_checkpoint_cannot_silently_load_as_legacy(tmp_path):
    from style_bert_vits2.models.utils.safetensors import load_safetensors, save_safetensors
    path = tmp_path / "enhanced.safetensors"
    save_safetensors(tiny_model(), 1, path)
    with pytest.raises(ValueError, match="use_prosody_adapter"):
        load_safetensors(path, tiny_model(False))


@pytest.mark.parametrize("zero_feature", ["style", "bert"])
def test_zero_placeholder_features_are_rejected(tmp_path, zero_feature):
    from data_utils import TextAudioSpeakerLoader
    from style_bert_vits2.models.hyper_parameters import HyperParametersData
    loader = TextAudioSpeakerLoader.__new__(TextAudioSpeakerLoader)
    loader.use_jp_extra = True
    loader.hparams = HyperParametersData(use_prosody_adapter=True)
    loader.spk_map = {"speaker": 0}
    bert = torch.zeros(1024, 3) if zero_feature == "bert" else torch.ones(1024, 3)
    loader.get_text = lambda *args: (None, bert, None, torch.ones(3, dtype=torch.long),
                                   torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long))
    loader.get_audio = lambda *args: (torch.ones(17, 20), torch.ones(1, 80))
    path = str(tmp_path / "audio.wav")
    np.save(path + ".npy", np.zeros(256) if zero_feature == "style" else np.ones(256))
    with pytest.raises(ValueError, match="Invalid/zero"):
        loader._load_sample([path, "speaker", "JP", "text", [], [], []])
