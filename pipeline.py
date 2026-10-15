"""Reusable, serialized PIPE owning official weights and per-request stream state."""
from __future__ import annotations

import gc
import json
import logging
import threading

import torch

from .frames import iter_prepared, target_size
from .memory import install_block_offload, install_decoder_offload, install_interrupt_checks, release_bundle
from .profiles import choose_profile, resolve_backend
from .tiling import blend_mask, tile_grid

log = logging.getLogger(__name__)
# Protect CUDA memory preparation and upstream's process-global seeding across pipes.
_GPU_LOCK = threading.RLock()


def interrupt():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


def prepare_device(device):
    import comfy.model_management as mm
    # FlashVSR+ owns unpatched upstream modules. Release Comfy-managed models first.
    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.set_device(device)


class FlashVSRPlusPipe:
    def __init__(self, *, mode, paths, device, profile="auto", attn_backend="auto",
                 offload="auto", decoder="tiny"):
        from .vendor.runtime.flashvsr_plus import build_inference_pipeline, stream_config_for_mode
        from .vendor.runtime.inference.config import DecoderConfig
        self.lock = threading.RLock()
        self.device, self.mode, self.decoder_kind = torch.device(device), mode, decoder
        self.profile_request, self.offload_request = profile, offload
        self.closed = False
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FlashVSR+ requires an NVIDIA CUDA GPU; CPU inference is not supported.")
        with _GPU_LOCK, torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The official FlashVSR+ preset requires CUDA BF16 support.")
            prepare_device(self.device)
            self.backend = resolve_backend(attn_backend, self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            self.initial_profile = choose_profile(total / 2**30, free / 2**30, self.backend, profile)
            # Build on CPU to avoid an initial all-weights CUDA spike. Resolve the
            # selected kernel against CUDA separately; the loader itself needs no GPU.
            config = stream_config_for_mode(mode, device="cpu", attn_kernel="sdpa", hardware_profile="generic")
            config.attn_kernel = self.backend
            if decoder == "vae":
                config.decoder = DecoderConfig(kind="vae", tiled=True)
            self.pipeline = build_inference_pipeline(config, paths)
            install_interrupt_checks(self.pipeline.bundle.denoiser, interrupt)
            config.device = str(self.device)
            config.validate()
            # Block offload can always be enabled by auto/OOM without reloading weights.
            self._block_adapter_installed = False
            self._decoder_offload_installed = False
            self._active_offload = None
            self._activate_offload(self.initial_profile.offload if offload == "auto" else offload)
            # A PIPE keeps weights in host RAM until its first inference request.
            release_bundle(self.pipeline.bundle)

    def _activate_offload(self, mode):
        if mode not in {"disabled", "model", "block"}:
            raise ValueError(f"Unknown offload mode: {mode}")
        # Once installed, the adapters are used for every subsequent request.
        # Automatic profiles only become more conservative during a PIPE's life.
        if self._block_adapter_installed:
            mode = "block"
        if mode == "block" and not self._block_adapter_installed:
            install_block_offload(self.pipeline.bundle.denoiser)
            self._block_adapter_installed = True
        if mode != "disabled" and not self._decoder_offload_installed:
            install_decoder_offload(self.pipeline.bundle.decoder, self.device,
                                    is_vae=self.decoder_kind == "vae")
            self._decoder_offload_installed = True
        self._active_offload = mode

    def _release(self):
        release_bundle(self.pipeline.bundle)
        gc.collect()
        torch.cuda.empty_cache()

    def close(self):
        with self.lock, _GPU_LOCK:
            if not self.closed:
                self._release()
                self.pipeline = None
                self.closed = True

    def _run_tile(self, images, height, width, tile, fps, progress):
        from .vendor.runtime.inference import InferenceRequest
        from .vendor.runtime.inference.preprocessing import TemporalVideoStream
        config = self.pipeline.config
        if config.decoder.decode_every == "auto":
            prepared = torch.cat(list(iter_prepared(images, height, width, 16, tile, interrupt)), dim=2)
            result = self.pipeline.infer(InferenceRequest(lq_video=prepared, fps=fps), progress=progress)
            return result.video, result.metadata
        # Same TemporalVideoStream -> StreamRunner as pipeline.infer's numeric
        # decode_every branch, with a bounded IMAGE source replacing OpenCV.
        source = iter_prepared(images, height, width,
                               config.decoder.decode_every * config.temporal_compression, tile, interrupt)
        stream = TemporalVideoStream(source, config.temporal_compression, 4)
        try:
            result = self.pipeline.runner.run(stream, None, progress=progress)
            value = result.video[:, :, :stream.source_frames]
            if value.shape[2] != images.shape[0]:
                raise RuntimeError(f"Official stream returned {value.shape[2]} frames for {images.shape[0]} inputs.")
            return value, {**result.metadata, "source_frames": stream.source_frames,
                           "output_frames": value.shape[2]}
        finally:
            stream.close()

    def _run_once(self, images, height, width, tiles, overlap, fps, progress):
        bundle = self.pipeline.bundle
        bundle.denoiser.to(self.device)
        if self._active_offload == "disabled":
            bundle.decoder.module.to(self.device)
        output = torch.zeros((images.shape[0], height, width, 3), dtype=torch.float32, device="cpu")
        weights = torch.zeros((height, width), dtype=torch.float32)
        metadata = {}
        for index, tile in enumerate(tiles):
            interrupt()
            progress(f"Spatial tile {index + 1}/{len(tiles)}")
            value, metadata = self._run_tile(images, height, width, tile, fps, progress)
            value = value.detach().to("cpu")[0].permute(1, 2, 3, 0)
            y, x, h, w = tile
            if tuple(value.shape) != (images.shape[0], h, w, 3):
                raise RuntimeError(f"Unexpected decoded shape: {tuple(value.shape)}")
            mask = blend_mask(tile, height, width, overlap)
            for start in range(0, images.shape[0], 4):
                clip = value[start:start + 4].float().add(1).mul_(0.5).clamp_(0, 1)
                output[start:start + 4, y:y + h, x:x + w].add_(clip * mask[None, :, :, None])
            weights[y:y + h, x:x + w].add_(mask)
            del value
        if not torch.all(weights > 0).item():
            raise RuntimeError("Spatial tiling left uncovered pixels.")
        for start in range(0, output.shape[0], 4):
            output[start:start + 4].div_(weights[None, :, :, None]).clamp_(0, 1)
        return output, metadata

    @torch.inference_mode()
    def upscale(self, images, *, scale=4.0, height=0, width=0, temporal_compression=4,
                tiled="auto", tile_size=512, tile_overlap=128, color_fix=False,
                color_correction_levels=5, seed=1, cache_offload="auto", decode_every=4,
                vae_tiled=True, vae_tile_size=256, vae_tile_overlap=64,
                oom_retry=True, fps=24.0, progress=None):
        from .vendor.runtime.inference.postprocessing import WaveletColorCorrector
        height, width = target_size(images, scale, height, width)
        if tiled not in {"auto", "enabled", "disabled"}:
            raise ValueError("tiled must be auto, enabled or disabled.")
        if temporal_compression not in (1, 2, 4):
            raise ValueError("temporal_compression must be 1, 2 or 4.")
        if decode_every < 0 or decode_every % 4:
            raise ValueError("decode_every must be 0 (auto) or a positive multiple of 4.")
        progress = progress or (lambda message: log.info("FlashVSR+: %s", message))
        with self.lock, _GPU_LOCK, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError("This PIPE was unloaded. Re-run the loader with a new reload_token.")
            prepare_device(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            profile = choose_profile(total / 2**30, free / 2**30, self.backend, self.profile_request)
            # Dense mask expansion makes full HD SDPA impractical even on 24GB.
            if (self.profile_request == "auto" and self.backend == "sdpa"
                    and height * width > 512 * 512):
                profile = choose_profile(total / 2**30, free / 2**30, self.backend, "low_vram")
            use_tiles = profile.tiled if tiled == "auto" else tiled == "enabled"
            size = profile.tile_size if tiled == "auto" else tile_size
            overlap = profile.tile_overlap if tiled == "auto" else tile_overlap
            mode = profile.offload if self.offload_request == "auto" else self.offload_request
            self._activate_offload(mode)
            config = self.pipeline.config
            config.seed = int(seed)
            config.temporal_compression = temporal_compression
            config.stream_cache_offload_mode = profile.cache_offload if cache_offload == "auto" else cache_offload
            config.stream_ffn_chunk_tokens = profile.ffn_chunk_tokens
            config.decoder.tiny_decoder_parallel_mode = profile.tiny_decoder_mode
            config.decoder.decode_every = (decode_every or "auto") if self.decoder_kind == "tiny" else "auto"
            if self.decoder_kind == "tiny":
                self.pipeline.bundle.decoder.parallel_mode = profile.tiny_decoder_mode
            else:
                if (vae_tile_size <= vae_tile_overlap or vae_tile_overlap < 0
                        or vae_tile_size < 128 or vae_tile_size % 8 or vae_tile_overlap % 8):
                    raise ValueError("VAE tile/overlap must be divisible by 8, with tile>=128 and overlap<tile.")
                decoder = self.pipeline.bundle.decoder
                decoder.tiled = bool(vae_tiled)
                decoder.tile_size = (vae_tile_size // 8,) * 2
                decoder.tile_stride = ((vae_tile_size - vae_tile_overlap) // 8,) * 2
            config.color_correction.enabled = bool(color_fix)
            config.color_correction.levels = color_correction_levels
            # pipeline.infer caches its corrector at construction; refresh it for
            # subsequent PIPE calls, including the VAE/full-timeline branch.
            self.pipeline.color_corrector = (WaveletColorCorrector(
                levels=color_correction_levels, chunk_size=4) if color_fix else None)
            config.validate()
            if use_tiles:
                tile_grid(height, width, size, overlap)  # validate before allocating
            attempts = []
            succeeded = False
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                for attempt in range(3):
                    tiles = tile_grid(height, width, size, overlap) if use_tiles else [(0, 0, height, width)]
                    exhausted = False
                    try:
                        output, metadata = self._run_once(images, height, width, tiles, overlap if use_tiles else 0, fps, progress)
                        metadata.update({"mode": self.mode, "profile": profile.to_dict(), "attn_backend": self.backend,
                                         "tiled": use_tiles, "tile_size": size if use_tiles else None,
                                         "tile_count": len(tiles), "offload": self._active_offload,
                                         "temporal_compression": temporal_compression, "seed": int(seed),
                                         "height": height, "width": width, "fps": fps,
                                         "oom_retries": attempts, "color_fix": color_fix,
                                         "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30})
                        succeeded = True
                        return output, json.dumps(metadata, ensure_ascii=False, indent=2)
                    except torch.cuda.OutOfMemoryError:
                        exhausted = not oom_retry or attempt == 2 or (use_tiles and size <= 256)
                    # Exit the exception scope before cleanup so traceback tensors
                    # cannot keep the failed attempt's CUDA allocations alive.
                    self._release()
                    if exhausted:
                        raise RuntimeError(
                            "FlashVSR+ exhausted CUDA memory. Use profile=low_vram, offload=block, "
                            "tile_size=256, tile_overlap=128, cache_offload=always and decode_every=4. "
                            "Close other GPU workloads or lower the requested resolution."
                        ) from None
                    size = min(512, max(256, (size // 2 // 128) * 128)) if use_tiles else 512
                    if self.backend == "sdpa":
                        size = min(size, 384)
                    overlap, use_tiles = 128, True
                    self._activate_offload("block")
                    config.stream_cache_offload_mode = "always"
                    config.stream_ffn_chunk_tokens = 4096
                    if self.decoder_kind == "tiny":
                        config.decoder.decode_every = 4
                        self.pipeline.bundle.decoder.parallel_mode = "serial"
                    attempts.append({"retry": attempt + 1, "tile_size": size})
                    progress(f"OOM recovery: restarting complete request with {size}px tiles and block offload")
                raise RuntimeError("FlashVSR+ produced no output")
            finally:
                # Also cleans up after interruption, OOM, or backend failure.
                if self._active_offload != "disabled" or not succeeded:
                    self._release()
