"""Explicit model layout, resumable downloads, and the V1 filename mapping."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import tempfile

PLUS_REPO = "ruchangyao/FlashVSR_plus"
PLUS_REVISION = "46b79520946ac67dec94d1566490690953c7debc"
V1_REPO = "JunhaoZhuang/FlashVSR-v1.1"
V1_REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"
log = logging.getLogger(__name__)


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


def _unique_roots(roots) -> list[Path]:
    return list(dict.fromkeys(Path(root).expanduser().resolve() for root in roots))


def default_model_roots() -> list[Path]:
    """Preserve configured priority, including the installation's own model folder."""
    import folder_paths
    key = "FlashVSR_Plus"
    local = Path(folder_paths.models_dir) / key
    # Desktop can pre-register an empty shared folder ahead of the user's models.
    # Always include the local default, even when this model type already exists.
    folder_paths.add_model_folder_path(key, str(local))
    return _unique_roots([*folder_paths.get_folder_paths(key), local])


def default_model_root() -> Path:
    return default_model_roots()[0]


def _atomic_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".flashvsr-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_models(root: Path, mode: str, decoder: str, auto_download: bool, *, search_roots=()):
    from .vendor.runtime.inference import ModelPaths
    roots = _unique_roots([root, *search_roots])
    root = roots[0]
    log.info("FlashVSR+ model search directories: %s", "; ".join(map(str, roots)))
    resolved = []
    context_root = root
    for item in model_files(mode, decoder):
        candidates = [(directory, directory / name) for directory in roots
                      for name in (item.local, *item.aliases)]
        found = next(((directory, path) for directory, path in candidates
                      if path.is_file() and path.stat().st_size), None)
        if found is None:
            checked = "\n".join(f"  {path}" for _, path in candidates)
            if not auto_download:
                raise FileNotFoundError(
                    f"Missing FlashVSR+ model: {item.local}\nSearched:\n{checked}\n"
                    f"Download {item.remote} from https://huggingface.co/{item.repo}/tree/{item.revision} "
                    "or enable auto_download. model_directory must contain turbo/base/v1, "
                    "not point to a mode subfolder."
                )
            path = root / item.local
            log.warning("FlashVSR+ local model missing: %s\nSearched:\n%s\nDownloading to: %s",
                        item.local, checked, path)
            from huggingface_hub import hf_hub_download
            # Hub manages resuming, verification, credentials, mirrors and download locking.
            downloaded = Path(hf_hub_download(item.repo, filename=item.remote, revision=item.revision))
            _atomic_copy(downloaded, path)
            directory = root
        else:
            directory, path = found
        if not resolved:
            context_root = directory
        log.info("FlashVSR+ using model: %s", path)
        resolved.append(str(path))
    contexts = [directory / "prompt_tensor" / "posi_prompt.pth" for directory in roots]
    context = next((path for path in contexts if path.is_file() and path.stat().st_size), None)
    if context is None:
        context = context_root / "prompt_tensor" / "posi_prompt.pth"
        _atomic_copy(Path(__file__).parent / "assets" / "posi_prompt.pth", context)
    return ModelPaths(dit=resolved[0], lq_projection=resolved[1], context=str(context),
                      tiny_decoder=resolved[2] if decoder == "tiny" else None,
                      vae=resolved[2] if decoder == "vae" else None)
