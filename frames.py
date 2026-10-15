"""Comfy IMAGE/VIDEO contracts and bounded official bicubic preprocessing."""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


def native_video_available():
    try:
        from comfy_api.latest import InputImpl, Types
        return hasattr(InputImpl, "VideoFromComponents") and hasattr(Types, "VideoComponents")
    except ImportError:
        return False


def unpack_input(images=None, video=None, audio=None, fps=24.0):
    if (images is None) == (video is None):
        raise ValueError("Connect exactly one of images (IMAGE) or video (VIDEO).")
    if video is not None:
        if not hasattr(video, "get_components"):
            raise TypeError("VIDEO must implement ComfyUI's get_components() interface.")
        components = video.get_components()
        images = components.images
        fps = float(components.frame_rate)
        audio = audio if audio is not None else components.audio
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("IMAGE must be [frames, height, width, 3 or 4], RGB in [0,1].")
    if not all(images.shape[:3]):
        raise ValueError("Empty image batches are not supported.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")
    return images, audio, fps


def target_size(images, scale, height, width):
    if bool(height) != bool(width):
        raise ValueError("Set both height and width, or leave both at 0 to use scale.")
    if height or width:
        if min(height, width) <= 0 or height % 128 or width % 128:
            raise ValueError("Explicit height/width must be positive multiples of 128.")
        return height, width
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    return tuple(max(128, math.ceil(value * scale / 128) * 128) for value in images.shape[1:3])


def iter_prepared(images, height, width, chunk_frames, tile=None, interrupt=lambda: None):
    from .vendor.runtime.inference.preprocessing.input import pil_center_crop_resize
    start = 0
    # Same first-window look-ahead as the official numeric decode_every path.
    window_size = chunk_frames + 1
    while start < images.shape[0]:
        interrupt()
        source = images[start:start + window_size, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(source).all().item():
            raise ValueError("Input images contain NaN or infinite values.")
        array = source.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        # Match CLI decoding: RGB8 -> PIL center crop -> BICUBIC -> [-1,1].
        array = pil_center_crop_resize(array, height, width, resample=Image.Resampling.BICUBIC)
        if tile is not None:
            y, x, h, w = tile
            array = array[:, y:y + h, x:x + w]
        yield (torch.from_numpy(np.ascontiguousarray(array)).permute(3, 0, 1, 2)
               .unsqueeze(0).float().div_(127.5).sub_(1))
        start += window_size
        window_size = chunk_frames


def create_video(images, fps, audio=None):
    from fractions import Fraction
    try:
        from comfy_api.latest import InputImpl, Types
    except ImportError as error:
        raise RuntimeError("This ComfyUI version has no native VIDEO API. Use VideoHelperSuite to save IMAGE frames.") from error
    return InputImpl.VideoFromComponents(Types.VideoComponents(
        images=images, audio=audio, frame_rate=Fraction(str(fps))))
