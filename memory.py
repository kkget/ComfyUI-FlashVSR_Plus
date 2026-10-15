"""Offload adapters for the official forward_stream calls (not nn.forward hooks)."""
from __future__ import annotations

import functools
import types

import torch


def ensure_cpu_output_fits(frames, height, width):
    """Reject impossible IMAGE results before allocating them or moving CUDA weights."""
    import psutil
    output_bytes = int(frames) * int(height) * int(width) * 3 * 4
    minimum_bytes = output_bytes + int(height) * int(width) * 4
    available = psutil.virtual_memory().available
    if minimum_bytes > available * .75:
        raise RuntimeError(
            f"FlashVSR+ IMAGE output needs at least {output_bytes / 2**30:.2f} GiB RAM "
            f"for {frames} frames at {width}x{height}; only {available / 2**30:.2f} GiB is available. "
            "Use FlashVSR+ Upscale VIDEO (Streaming) to write video without a full IMAGE batch, "
            "or reduce the output dimensions/frame count. GPU tiling/offload cannot reduce "
            "the RAM required by the complete IMAGE output."
        )
    return output_bytes


def install_interrupt_checks(denoiser, check):
    for block in denoiser.blocks:
        original = block.forward_stream

        def wrapped(module, *args, _original=original, **kwargs):
            check()
            return _original(*args, **kwargs)

        block.forward_stream = types.MethodType(wrapped, block)


def install_block_offload(denoiser):
    """Keep only the active transformer block on CUDA, including during restore_dit."""
    original_to = denoiser.to

    def move(module, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        # Upstream invokes denoiser.to(device) after every incremental decode.
        # Intercept that call so it cannot accidentally load all 30 blocks.
        if isinstance(target, (str, torch.device, int)):
            device = torch.device("cuda", target) if isinstance(target, int) else torch.device(target)
            if device.type == "cuda":
                for name, child in module.named_children():
                    if name != "blocks":
                        child.to(*args, **kwargs)
                return module
        return original_to(*args, **kwargs)

    denoiser.to = types.MethodType(move, denoiser)
    for block in denoiser.blocks:
        original = block.forward_stream

        def wrapped(module, x, *args, _original=original, **kwargs):
            module.to(x.device)
            try:
                return _original(x, *args, **kwargs)
            finally:
                module.to("cpu")

        block.forward_stream = types.MethodType(wrapped, block)


def install_decoder_offload(decoder, device: torch.device, *, is_vae: bool):
    """Move decoder weights only for decoding; release them while DiT runs."""
    for name in ("decode", "decode_stream_segment"):
        original = getattr(decoder, name, None)
        if original is None:
            continue

        @functools.wraps(original)
        def wrapped(*args, _original=original, **kwargs):
            decoder.module.to(device)
            try:
                if is_vae and not decoder.tiled:
                    args = (args[0].to(device), *args[1:])
                return _original(*args, **kwargs)
            finally:
                decoder.module.to("cpu")

        setattr(decoder, name, wrapped)


def release_bundle(bundle):
    projection = getattr(bundle.denoiser, "LQ_proj_in", None)
    if projection is not None:
        projection.clear_cache()
    bundle.decoder.reset()
    bundle.denoiser.to("cpu")
    bundle.decoder.module.to("cpu")
