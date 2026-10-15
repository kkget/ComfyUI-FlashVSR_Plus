"""Explicit model layout, resumable downloads, and the V1 filename mapping."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile

PLUS_REPO = "ruchangyao/FlashVSR_plus"
PLUS_REVISION = "46b79520946ac67dec94d1566490690953c7debc"
V1_REPO = "JunhaoZhuang/FlashVSR-v1.1"
V1_REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"


@dataclass(frozen=True)
class ModelFile:
    local: str
    remote: str
    repo: str = PLUS_REPO
    revision: str = PLUS_REVISION
    aliases: tuple[str, ...] = ()


def model_files(mode: str, decoder: str = "tiny") -> list[ModelFile]:
    if mode not in {"turbo", "base", "v1"}:
        raise ValueError(f"Unknown mode: {mode}")
    if decoder not in {"tiny", "vae"}:
        raise ValueError(f"Unknown decoder: {decoder}")
    if mode == "v1":
        files = [
            ModelFile("v1/dit.safetensors", "diffusion_pytorch_model_streaming_dmd.safetensors",
                      V1_REPO, V1_REVISION, ("v1/diffusion_pytorch_model_streaming_dmd.safetensors",)),
            ModelFile("v1/LQ_proj_in.ckpt", "LQ_proj_in.ckpt", V1_REPO, V1_REVISION,
                      ("v1/lq_proj_in.safetensors",)),
        ]
    else:
        files = [ModelFile(f"{mode}/{name}", f"{mode}/{name}")
                 for name in ("dit.safetensors", "lq_proj_in.safetensors")]
    files.append(ModelFile("TCDecoder.ckpt", "TCDecoder.ckpt") if decoder == "tiny"
                 else ModelFile("Wan2.1_VAE.pth", "Wan2.1_VAE.pth"))
    return files


def default_model_root() -> Path:
    import folder_paths
    key = "FlashVSR_Plus"
    if key not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(key, str(Path(folder_paths.models_dir) / key))
    return Path(folder_paths.get_folder_paths(key)[0])


def _atomic_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".flashvsr-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_models(root: Path, mode: str, decoder: str, auto_download: bool):
    from .vendor.runtime.inference import ModelPaths
    resolved = []
    for item in model_files(mode, decoder):
        candidates = [root / item.local, *(root / alias for alias in item.aliases)]
        path = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
        if path is None:
            if not auto_download:
                raise FileNotFoundError(
                    f"Missing FlashVSR+ model: {candidates[0]}\n"
                    f"Download {item.remote} from https://huggingface.co/{item.repo}/tree/{item.revision} "
                    "or enable auto_download."
                )
            from huggingface_hub import hf_hub_download
            # Hub manages resuming, verification, credentials, mirrors and download locking.
            downloaded = Path(hf_hub_download(item.repo, filename=item.remote, revision=item.revision))
            path = candidates[0]
            _atomic_copy(downloaded, path)
        resolved.append(str(path))
    context = root / "prompt_tensor" / "posi_prompt.pth"
    if not context.is_file():
        _atomic_copy(Path(__file__).parent / "assets" / "posi_prompt.pth", context)
    return ModelPaths(dit=resolved[0], lq_projection=resolved[1], context=str(context),
                      tiny_decoder=resolved[2] if decoder == "tiny" else None,
                      vae=resolved[2] if decoder == "vae" else None)
