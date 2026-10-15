"""Select a policy from physical capacity, currently free VRAM and usable kernels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import logging

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    total_gib: float
    free_gib: float
    backend: str
    tiled: bool
    tile_size: int
    tile_overlap: int
    offload: str
    cache_offload: str
    decode_every: int
    tiny_decoder_mode: str
    ffn_chunk_tokens: int
    reason: str

    def to_dict(self):
        return asdict(self)


def choose_profile(total_gib: float, free_gib: float, backend: str, profile: str = "auto") -> MemoryProfile:
    if profile not in {"auto", "official", "low_vram"}:
        raise ValueError(f"Invalid profile: {profile}")
    # A nominal 24 GB GPU may report 23.7–23.9 GiB, particularly under Windows/WDDM.
    large = total_gib >= 23.0
    enough_free = free_gib >= 18.0
    name = ("official" if large and enough_free else "low_vram") if profile == "auto" else profile
    low = name == "low_vram"
    tile = 512 if free_gib >= 8 else (384 if free_gib >= 5 else 256)
    if backend == "sdpa":
        tile = min(tile, 384)  # SDPA materializes the sparse mask.
    reason = (f"{total_gib:.2f} GiB total, {free_gib:.2f} GiB free; "
              + ("automatic capacity/free-memory policy" if profile == "auto" else "manual override"))
    return MemoryProfile(name, total_gib, free_gib, backend, low, tile, 128,
                         "block" if low else "model", "always" if low else "auto",
                         4, "serial" if low else "auto", 8192 if low else 32768, reason)


def probe_backend(name: str, device: torch.device):
    """Check import AND a real tiny CUDA call, catching incompatible binary/JIT installs."""
    if name == "sdpa":
        return
    if name == "triton_block_sparse":
        module = importlib.import_module(".vendor.runtime.flashvsr_attn", __package__)
        function = module.stream_block_sparse_attention
    else:
        importlib.import_module("block_sparse_attn")
        from .vendor.runtime.models.attention_backends.kernels import _run_block_sparse
        function = None
    with torch.inference_mode(), torch.cuda.device(device):
        q = torch.randn((1, 1, 2, 128, 128), device=device, dtype=torch.bfloat16)
        mask = torch.ones((1, 1, 2, 2), device=device, dtype=torch.bool)
        out = (function(q, q, q, mask) if function is not None else
               _run_block_sparse(q.flatten(2, 3), q.flatten(2, 3), q.flatten(2, 3), mask))
        if not torch.isfinite(out).all().item():
            raise RuntimeError(f"{name} returned nonfinite values")
        torch.cuda.synchronize(device)


def resolve_backend(requested: str, device: torch.device):
    if requested not in {"auto", "block_sparse", "triton_block_sparse", "sdpa"}:
        raise ValueError(f"Unsupported attention backend: {requested}")
    candidates = ("block_sparse", "triton_block_sparse", "sdpa") if requested == "auto" else (requested,)
    for candidate in candidates:
        try:
            probe_backend(candidate, device)
            return candidate
        except Exception as error:
            if requested != "auto":
                raise RuntimeError(f"FlashVSR+ backend {candidate} failed its CUDA check: {error}") from error
            log.info("FlashVSR+ skipping %s: %s", candidate, error)
    raise RuntimeError("No usable attention backend")
