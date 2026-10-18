"""Full-size official-pretrain/data/trainer smoke on Modal, not a quality claim.

Uses the user's existing af_alt_20file BERT/style caches read-only, copies only
that dataset into the dedicated test Volume, and verifies official G/D hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import TextAudioSpeakerCollate, TextAudioSpeakerLoader
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_jp_extra import MultiPeriodDiscriminator, SynthesizerTrn
from style_bert_vits2.nlp.symbols import SYMBOLS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_real_features(root, wav_dir, records, hps):
    """Regenerate features: the historical benchmark cache contained zeros."""
    metadata_path = root / "real_features_v1.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models, clean_text, extract_bert_feature

    bert_repo = "ku-nlp/deberta-v2-large-japanese-char-wwm"
    bert_models.load_model(Languages.JP, bert_repo, device_map="cuda")
    bert_models.load_tokenizer(Languages.JP, bert_repo)
    metadata = {}
    for fields in records:
        name = fields[0].replace("\\", "/").split("/")[-1]
        norm, phones, tones, word2ph = clean_text(fields[3], Languages.JP, use_jp_extra=True)
        aligned = [value * 2 for value in word2ph] if hps.data.add_blank else word2ph.copy()
        if hps.data.add_blank:
            aligned[0] += 1
        features = extract_bert_feature(norm, aligned, Languages.JP, "cuda")
        assert features.abs().sum() > 0
        torch.save(features.cpu(), wav_dir / name.replace(".wav", ".bert.pt"))
        metadata[name] = [norm, " ".join(phones), " ".join(map(str, tones)), " ".join(map(str, word2ph))]
    bert_models.unload_all_models()
    bert_models.unload_all_tokenizers()
    print("Generated actual Japanese BERT features", flush=True)
    # Only official model files are loaded by this preparation routine.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    from pyannote.audio import Inference, Model
    speaker = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(speaker, window="whole")
    inference.to(torch.device("cuda"))
    for name in metadata:
        vector = np.asarray(inference(str(wav_dir / name)), dtype=np.float32).reshape(256)
        assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
        np.save(wav_dir / (name + ".npy"), vector)
    del inference, speaker
    torch.cuda.empty_cache()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print("Generated actual WeSpeaker style embeddings", flush=True)
    return metadata


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    root = Path("/cache/smoke")
    root.mkdir(parents=True, exist_ok=True)
    reference = Path("/reference/Style-Bert-VITS2/Data/af_alt_20file")
    model_repo = "litagin/Style-Bert-VITS2-2.0-base-JP-Extra"
    info = HfApi().model_info(model_repo, files_metadata=True)
    weights = {}
    hashes = {}
    for filename in ("G_0.safetensors", "D_0.safetensors"):
        expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == filename)
        candidate = reference / "models" / filename
        if not candidate.exists() or sha256(candidate) != expected:
            candidate = Path(hf_hub_download(model_repo, filename))
        assert sha256(candidate) == expected
        weights[filename] = candidate
        hashes[filename] = expected
    print("Verified official G/D pretrain hashes", flush=True)

    # Confirm the prepared cache has the same 20 identities as the supplied HF archive.
    archive = hf_hub_download("einiite/af_alt_20file", "af_alt_20file.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as z:
        audio_names = {Path(name).name for name in z.namelist() if name.endswith(".wav")}
    assert len(audio_names) == 20
    wav_dir = root / "wavs"
    wav_dir.mkdir(exist_ok=True)
    for name in audio_names:
        for filename in (name, name + ".npy", name.replace(".wav", ".bert.pt")):
            destination = wav_dir / filename
            if not destination.exists():
                shutil.copyfile(reference / "wavs" / filename, destination)
    hps = HyperParameters.load_from_json(reference / "config.json")
    hps.model.use_prosody_adapter = True
    hps.model.use_wavlm_discriminator = False
    hps.model.use_duration_discriminator = False
    hps.data.use_prosody_adapter = True
    hps.data.use_jp_extra = True
    hps.train.bf16_run = True
    hps.train.fp16_run = False
    hps.train.batch_size = 2
    hps.train.eval_interval = 1000000
    hps.train.log_interval = 1
    hps.speedup = False
    hps.model_dir = str(root / "models")
    Path(hps.model_dir).mkdir(exist_ok=True)

    source_records = []
    for split in ("train", "val"):
        for line in (reference / f"{split}.list").read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                source_records.append(line.split("|"))
    real_features = prepare_real_features(root, wav_dir, source_records, hps)

    records = []
    for split in ("train", "val"):
        output = []
        for line in (reference / f"{split}.list").read_text(encoding="utf-8-sig").splitlines():
            fields = line.split("|")
            assert len(fields) == 7
            name = fields[0].replace("\\", "/").split("/")[-1]
            assert name in audio_names
            fields[0] = str(wav_dir / name)
            fields[3:] = real_features[name]
            output.append("|".join(fields))
            records.append(name)
        path = root / f"{split}.list"
        if split == "val" and not output:
            training_path = root / "train.list"
            training_records = training_path.read_text(encoding="utf-8").splitlines()
            output = training_records[-2:]
            training_path.write_text("\n".join(training_records[:-2]) + "\n", encoding="utf-8")
        path.write_text("\n".join(output) + "\n", encoding="utf-8")
        setattr(hps.data, f"{'training' if split == 'train' else 'validation'}_files", str(path))
    assert set(records) == audio_names

    dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    validation = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    collate = TextAudioSpeakerCollate(use_jp_extra=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
    first = next(iter(loader))
    validation_batch = collate([validation[0]])
    print(f"Extracted/cached prosody: {len(dataset)} train, {len(validation)} validation", flush=True)

    parameters = dict(
        n_vocab=len(SYMBOLS), spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers, **hps.model.model_dump(),
    )
    state = load_file(str(weights["G_0.safetensors"]))
    state.pop("iteration", None)
    parameters["use_prosody_adapter"] = False
    legacy = SynthesizerTrn(**parameters).cuda().eval()
    result = legacy.load_state_dict(state, strict=False)
    expected_missing = {"enc_p.style_proj.weight", "enc_p.style_proj.bias", "emb_g.weight"}
    assert set(result.missing_keys) <= expected_missing and not result.unexpected_keys, result
    official_missing = result.missing_keys
    # The official base omits speaker/style-specific parameters by design. Use
    # the SAME standard initialization on both sides of the compatibility test.
    for key in official_missing:
        state[key] = legacy.state_dict()[key].detach().cpu()
    x, lengths, spec, spec_lengths, y, y_lengths, sid, tone, lang, bert, style, targets = validation_batch
    # A style preset, rather than the held-out recording's style embedding.
    mean_style = torch.stack([sample[7] for sample in dataset]).mean(0, keepdim=True).cuda()
    infer_args = (x.cuda(), lengths.cuda(), sid.cuda(), tone.cuda(), lang.cuda(), bert.cuda(), mean_style)
    with torch.no_grad():
        torch.manual_seed(7)
        before = legacy.infer(*infer_args, noise_scale=0, noise_scale_w=0)[0].cpu()
    del legacy
    torch.cuda.empty_cache()
    parameters["use_prosody_adapter"] = True
    model = SynthesizerTrn(**parameters).cuda()
    result = model.load_state_dict(state, strict=False)
    assert all(k.startswith("prosody_adapter.") for k in result.missing_keys)
    assert not result.unexpected_keys
    del state
    model.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        migrated = model.infer(*infer_args, noise_scale=0, noise_scale_w=0)[0].cpu()
    initial_difference = float((before - migrated).abs().max())
    torch.testing.assert_close(before, migrated, rtol=1e-5, atol=2e-6)
    print("Full-size official checkpoint migration max absolute difference:", initial_difference, flush=True)

    discriminator = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
    dstate = load_file(str(weights["D_0.safetensors"]))
    dstate.pop("iteration", None)
    discriminator.load_state_dict(dstate)
    del dstate
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29591", rank=0, world_size=1)
    generator = DDP(model, device_ids=[0])
    discriminator = DDP(discriminator, device_ids=[0])
    discriminator._sbv2_freeze_for_generator = False
    optim_g = torch.optim.AdamW(model.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    event_dir = root / f"events_{time.time_ns()}"
    writer = SummaryWriter(str(event_dir))
    import train_ms_jp_extra as trainer
    trainer.global_step = 1
    start = time.monotonic()
    trainer.train_and_evaluate(
        0, 0, 1, hps, (generator, discriminator, None, None, None),
        (optim_g, optim_d, None, None), (None, None, None, None),
        torch.cuda.amp.GradScaler(enabled=False), (loader, None), trainer.logger,
        (writer, writer), None, 0, algorithm_optimizations=True,
    )
    elapsed = time.monotonic() - start
    writer.close()
    print("Style norms:", [float(sample[7].norm()) for sample in dataset], flush=True)
    print("Duration diagnostics:", {
        "weight_norm": float(model.prosody_adapter.duration_style.weight.norm()),
        "bias_norm": float(model.prosody_adapter.duration_style.bias.norm()),
        "weight_grad_norm": float(model.prosody_adapter.duration_style.weight.grad.norm()),
        "bias_grad_norm": float(model.prosody_adapter.duration_style.bias.grad.norm()),
        "dp_proj_norm": float(model.dp.proj.weight.norm()),
        "sdp_proj_norm": float(model.sdp.proj.weight.norm()),
        "learning_rate": hps.train.learning_rate,
    }, flush=True)
    assert model.prosody_adapter.prior_residual.weight.norm().item() > 0
    assert model.prosody_adapter.duration_style.weight.norm().item() > 0
    assert model.prosody_adapter.duration_gate.abs().item() > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        audio = model.infer(*infer_args, noise_scale=0, noise_scale_w=0)[0]
    assert torch.isfinite(audio).all()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(event_dir)).Reload()
    prosody_losses = [event.value for event in events.Scalars("loss/g/prosody")]
    assert len(prosody_losses) == len(loader) and np.isfinite(prosody_losses).all()
    from style_bert_vits2.models.utils.safetensors import save_safetensors
    save_safetensors(model, trainer.global_step, root / "smoke_adapter.safetensors", for_infer=True)
    (root / "smoke_config.json").write_text(hps.model_dump_json(indent=2), encoding="utf-8")
    summary = {
        "purpose": "Functional smoke test; not an evaluation of acting quality",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "dataset": "einiite/af_alt_20file", "prepared_samples": len(records),
        "features": "Actual Japanese DeBERTa + WeSpeaker embeddings, regenerated from data",
        "train_samples": len(dataset), "validation_samples": len(validation),
        "official_pretrain": model_repo, "pretrain_sha256": hashes,
        "official_missing_speaker_style_parameters": official_missing,
        "initial_output_matches_fp32_tolerance": True,
        "initial_output_max_abs_difference": initial_difference, "precision": "bf16", "wavlm": False,
        "training_steps": len(loader), "training_seconds": elapsed,
        "prosody_losses": prosody_losses,
        "prior_residual_norm": model.prosody_adapter.prior_residual.weight.norm().item(),
        "duration_style_norm": model.prosody_adapter.duration_style.weight.norm().item(),
        "duration_gate": model.prosody_adapter.duration_gate.item(),
        "inference_finite": True,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
