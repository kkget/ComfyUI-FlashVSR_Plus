"""Legacy node registration also works in current ComfyUI; VIDEO is feature-detected."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from .frames import create_video, native_video_available, target_size, unpack_input
from .models import default_model_roots, ensure_models

CATEGORY = "FlashVSR+"
PIPE = "FLASHVSR_PLUS_PIPE"
log = logging.getLogger(__name__)


class FlashVSRPlusLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["turbo", "base", "v1"], {"default": "turbo"}),
            "profile": (["auto", "official", "low_vram"], {"default": "auto"}),
            "attn_backend": (["auto", "block_sparse", "triton_block_sparse", "sdpa"],),
            "offload": (["auto", "model", "block", "disabled"],),
            "decoder": (["tiny", "vae"], {"default": "tiny"}),
            "auto_download": ("BOOLEAN", {"default": True,
                                         "tooltip": "Download missing files only. False: strictly local, no network."}),
            "model_directory": ("STRING", {"default": "", "tooltip":
                "Empty: search all FlashVSR_Plus model paths and ComfyUI/models/FlashVSR_Plus. "
                "Explicit: only this root (containing turbo/base/v1)."}),
            "reload_token": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                                      "tooltip": "Change after explicitly unloading a cached PIPE."}),
        }}

    RETURN_TYPES = (PIPE, "STRING")
    RETURN_NAMES = ("pipe", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load official FlashVSR+ once. Auto selects full spatial inference or low-VRAM tiles."

    def load(self, mode, profile, attn_backend, offload, decoder, auto_download, model_directory, reload_token=0):
        import comfy.model_management as mm
        import torch
        from .pipeline import FlashVSRPlusPipe
        device = mm.get_torch_device()
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FlashVSR+ requires an NVIDIA CUDA GPU.")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("FlashVSR+ requires CUDA BF16 support.")
        roots = ([Path(model_directory.strip()).expanduser().resolve()]
                 if model_directory.strip() else default_model_roots())
        paths = ensure_models(roots[0], mode, decoder, auto_download, search_roots=roots[1:])
        pipe = FlashVSRPlusPipe(mode=mode, paths=paths, device=device,
                               profile=profile, attn_backend=attn_backend, offload=offload, decoder=decoder)
        return pipe, json.dumps(pipe.initial_profile.to_dict(), indent=2)


class FlashVSRPlusUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"images": ("IMAGE",), "audio": ("AUDIO",)}
        if native_video_available():
            optional["video"] = ("VIDEO",)
        return {"required": {
            "pipe": (PIPE,),
            "scale": ("FLOAT", {"default": 4.0, "min": 0.25, "max": 8.0, "step": 0.25}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 128}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 128}),
            "temporal_compression": ([4, 2, 1],),
            "tiled": (["auto", "enabled", "disabled"],),
            "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 128}),
            "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 128}),
            "color_fix": ("BOOLEAN", {"default": False}),
            "color_correction_levels": ("INT", {"default": 5, "min": 1, "max": 8}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            "cache_offload": (["auto", "always", "never"],),
            "decode_every": ("INT", {"default": 4, "min": 0, "max": 128, "step": 4,
                                      "tooltip": "Latent frames per decode, 0=full timeline. TinyDecoder only."}),
            "vae_tiled": ("BOOLEAN", {"default": True}),
            "vae_tile_size": ("INT", {"default": 256, "min": 128, "max": 2048, "step": 8}),
            "vae_tile_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
            "oom_retry": ("BOOLEAN", {"default": True}),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
        }, "optional": optional}

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", PIPE, "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "pipe", "info")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Connect IMAGE or native VIDEO. Return exactly the source frame count on CPU; preserve audio/FPS."

    def upscale(self, pipe, images=None, video=None, audio=None, fps=24.0, **kwargs):
        # File-backed VIDEO can expose dimensions without decoding the full clip.
        if images is None and video is not None and all(hasattr(video, method) for method in
                                                        ("get_dimensions", "get_frame_count")):
            from .memory import ensure_cpu_output_fits
            width, height = video.get_dimensions()
            frames = video.get_frame_count()
            if frames > 0:
                shape = SimpleNamespace(shape=(frames, height, width, 3))
                height, width = target_size(shape, kwargs.get("scale", 4.0),
                                            kwargs.get("height", 0), kwargs.get("width", 0))
                ensure_cpu_output_fits(frames, height, width)
        images, audio, fps = unpack_input(images=images, video=video, audio=audio, fps=fps)
        import comfy.model_management as mm
        def progress(message):
            mm.throw_exception_if_processing_interrupted()
            log.info("FlashVSR+: %s", message)
        output, info = pipe.upscale(images, fps=fps, progress=progress, **kwargs)
        return output, audio, fps, pipe, info


class FlashVSRPlusVideoUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        required = dict(FlashVSRPlusUpscale.INPUT_TYPES()["required"])
        required.pop("fps")
        required["scale"] = ("FLOAT", {"default": 2.0, "min": .25, "max": 8.0, "step": .25})
        required["decode_every"] = ("INT", {"default": 4, "min": 4, "max": 128, "step": 4})
        return {"required": {"video": ("VIDEO",), **required,
                             "crf": ("INT", {"default": 18, "min": 0, "max": 40}),
                             "keep_audio": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("VIDEO", PIPE, "STRING")
    RETURN_NAMES = ("video", "pipe", "info")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Write long VIDEO with bounded RAM using the official continuous temporal runner. "
                   "Requires tiny decoder; spatial tiles use temporary RGB8 disk storage. Connect to SaveVideo.")

    def upscale(self, pipe, video, crf=18, keep_audio=True, **kwargs):
        import tempfile
        import uuid
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        from .video_streaming import VideoFileSource, VideoFileSink
        if pipe.decoder_kind != "tiny" or kwargs.get("decode_every", 4) < 4:
            raise ValueError("Streaming VIDEO requires Loader decoder=tiny and decode_every>=4.")

        def progress(message):
            mm.throw_exception_if_processing_interrupted()
            log.info("FlashVSR+: %s", message)

        temporary = Path(folder_paths.get_temp_directory())
        temporary.mkdir(parents=True, exist_ok=True)
        destination = temporary / f"flashvsr-plus-{uuid.uuid4().hex}.mp4"
        try:
            with tempfile.TemporaryDirectory(prefix="flashvsr-source-", dir=temporary) as work:
                source_path = Path(work) / "source.mkv"
                progress("Preparing file-backed VIDEO input")
                # Current native file VIDEO remuxes or transcodes this incrementally,
                # applying upstream trim/crop without get_components() materialization.
                video.save_to(str(source_path))
                source = VideoFileSource(source_path)
                sink = VideoFileSink(destination, temporary, crf=crf, keep_audio=keep_audio)
                path, info = pipe.upscale(source, fps=source.fps, progress=progress,
                                          _video_sink=sink, **kwargs)
            return InputImpl.VideoFromFile(path), pipe, info
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


class FlashVSRPlusToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "forceInput": True})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, images, fps, audio=None):
        return (create_video(images, fps, audio),)


class FlashVSRPlusUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": (PIPE,)}, "optional": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Release PIPE host/CUDA weights. Feed the upscaled images to enforce execution order."

    def unload(self, pipe, images=None):
        pipe.close()
        if images is None:
            import torch
            images = torch.zeros((1, 1, 1, 3))
        return (images,)


class FlashVSRPlusDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profile": (["auto", "official", "low_vram"],),
                             "attn_backend": (["auto", "block_sparse", "triton_block_sparse", "sdpa"],)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def inspect(self, profile="auto", attn_backend="auto"):
        import torch
        import comfy.model_management as mm
        from .profiles import choose_profile, resolve_backend
        from .pipeline import _GPU_LOCK
        device = mm.get_torch_device()
        if device.type != "cuda":
            result = {"device": str(device), "error": "FlashVSR+ requires CUDA"}
        else:
            with _GPU_LOCK, torch.cuda.device(device):
                backend = resolve_backend(attn_backend, device)
                free, total = torch.cuda.mem_get_info(device)
                result = {"device": torch.cuda.get_device_name(device),
                          **choose_profile(total / 2**30, free / 2**30, backend, profile).to_dict(),
                          "native_video": native_video_available()}
        info = json.dumps(result, ensure_ascii=False, indent=2)
        return {"ui": {"text": [info]}, "result": (info,)}


NODE_CLASS_MAPPINGS = {
    "FlashVSRPlusLoader": FlashVSRPlusLoader,
    "FlashVSRPlusUpscale": FlashVSRPlusUpscale,
    "FlashVSRPlusUnload": FlashVSRPlusUnload,
    "FlashVSRPlusDiagnostics": FlashVSRPlusDiagnostics,
}
if native_video_available():
    NODE_CLASS_MAPPINGS["FlashVSRPlusToVideo"] = FlashVSRPlusToVideo
    NODE_CLASS_MAPPINGS["FlashVSRPlusVideoUpscale"] = FlashVSRPlusVideoUpscale

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRPlusLoader": "FlashVSR+ Load PIPE (Auto VRAM)",
    "FlashVSRPlusUpscale": "FlashVSR+ Upscale (IMAGE / VIDEO)",
    "FlashVSRPlusVideoUpscale": "FlashVSR+ Upscale VIDEO (Streaming)",
    "FlashVSRPlusToVideo": "FlashVSR+ Create VIDEO",
    "FlashVSRPlusUnload": "FlashVSR+ Unload PIPE",
    "FlashVSRPlusDiagnostics": "FlashVSR+ GPU / Backend Check",
}
