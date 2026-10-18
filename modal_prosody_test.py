"""Run validation on Modal: modal run modal_prosody_test.py --mode test."""
from pathlib import Path

import modal

app = modal.App("sbv2-expressive-prosody-validation")
volume = modal.Volume.from_name("sbv2-prosody-20261019", create_if_missing=True)
reference = modal.Volume.from_name("vits-storage")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0", "numpy==1.26.4",
        "scipy", "librosa==0.10.2.post1", "soundfile", "pyworld-prebuilt",
        "numba", "pydantic>=2", "loguru", "safetensors", "pyyaml",
        "transformers==4.46.3", "huggingface_hub", "tensorboard", "pytest", "ruff",
    )
    .pip_install("git+https://github.com/supertone-oss-archive/super-monotonic-align.git")
    .pip_install("onnxruntime", "pyopenjtalk-dict", "num2words", "sentencepiece", "accelerate", "jaconv")
    .pip_install("pyannote.audio==3.3.2", "torch==2.6.0", "torchaudio==2.6.0", "numpy==1.26.4")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/repo"})
    .add_local_dir(
        Path(__file__).parent, remote_path="/repo",
        ignore=[".git", "bert", "slm", "model_assets", "Data", "pretrained",
                "pretrained_jp_extra", "__pycache__", "*.pyc"],
    )
)


@app.function(image=image, volumes={"/cache": volume, "/reference": reference},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def probe():
    import json
    import os
    from huggingface_hub import HfApi
    keys = [k for k in os.environ if "HUGG" in k.upper() or k.startswith("HF_")]
    print("Hugging Face environment variable names:", keys)
    token = next((os.environ[k] for k in (
        "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_ACCESS_TOKEN", "HUGGING_FACE_TOKEN", "huggingface") if os.environ.get(k)), None)
    api = HfApi(token=token)
    info = api.dataset_info("einiite/af_alt_20file", files_metadata=True)
    result = {"dataset_files": [{"name": f.rfilename, "bytes": f.size} for f in info.siblings]}
    from huggingface_hub import hf_hub_download
    import zipfile
    archive = hf_hub_download("einiite/af_alt_20file", "af_alt_20file.zip", repo_type="dataset", token=token)
    with zipfile.ZipFile(archive) as z:
        result["archive_files"] = z.namelist()
        result["text_samples"] = {}
        for name in z.namelist():
            if name.endswith((".txt", ".list")):
                raw = z.read(name)
                encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
                result["text_samples"][name] = raw.decode(encoding, errors="replace")[:1200]
    root = Path("/reference/Style-Bert-VITS2")
    prepared = root / "Data/af_alt_20file"
    result["prepared_files"] = [str(p.relative_to(prepared)) for p in prepared.rglob("*") if p.is_file()][:35]
    with zipfile.ZipFile(root / "Data/af_alt_20file__wavs.zip") as z:
        result["prepared_zip_files"] = z.namelist()[:15]
    with zipfile.ZipFile(root / "pth/model_e200.zip") as z:
        result["pretrain_zip_files"] = z.namelist()[:15]
    print(json.dumps(result, ensure_ascii=False))
    volume.commit()
    return result


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=16384,
              volumes={"/cache": volume, "/reference": reference}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3600)
def validate(mode: str):
    import os
    import subprocess
    os.chdir("/repo")
    if os.environ.get("huggingface") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["huggingface"]
    if mode == "test":
        command = ["python", "-u", "-m", "pytest", "tests/test_prosody.py", "-v", "-s", "-ra"]
    elif mode == "smoke":
        command = ["python", "tests/prosody_smoke.py"]
    else:
        raise ValueError(mode)
    print("Starting Modal validation:", mode, flush=True)
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    exit_code = process.wait()
    volume.commit()
    return {"exit_code": exit_code, "output": "".join(lines)}


@app.function(image=image, timeout=300)
def check_package(patch: bytes):
    import py_compile
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "repo"
        subprocess.run([
            "git", "clone", "--quiet", "--depth", "1", "--single-branch",
            "--branch", "260711fp16_bf_comm_datautils_losses_h200ultra_spd",
            "https://github.com/fynyte/Style-BERT-VITS2-GeneLab-Blackwell.git", str(target),
        ], check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
        if revision != "84eebb27d893982ddc6e1e196c1f6dda0b1ad399":
            raise RuntimeError("Upstream moved; verify the patch against its recorded base")
        for arguments in (["--check", "--whitespace=nowarn"], ["--whitespace=nowarn"]):
            result = subprocess.run(["git", "apply", *arguments, "-"], input=patch,
                                    cwd=target, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                return {"exit_code": result.returncode, "output": result.stdout.decode()}
        for line in patch.decode().splitlines():
            if line.startswith("+++ b/") and line.endswith(".py"):
                py_compile.compile(str(target / line[6:]), doraise=True)
        return {"exit_code": 0, "base_commit": revision,
                "output": "Patch applies to clean upstream; modified Python files compile"}


@app.local_entrypoint()
def main(mode: str = "test", output: str = "", patch: str = ""):
    import json
    if mode == "package":
        result = check_package.remote(Path(patch).read_bytes())
    else:
        result = probe.remote() if mode == "probe" else validate.remote(mode)
    if output:
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.get("exit_code", 0):
        raise RuntimeError("Modal validation failed; see output above")
