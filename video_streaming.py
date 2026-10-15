"""Continuous official temporal inference with RGB8 tile spools and bounded RAM.

Each spatial tile retains its own official KV/decoder state across the complete
clip. Completed decode segments go to disk instead of a full IMAGE tensor.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch

from .memory import ensure_cpu_output_fits
from .tiling import blend_mask


class VideoFileSource:
    def __init__(self, path):
        self.path = Path(path)
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {self.path}")
            width, height, frames = (int(capture.get(key)) for key in
                                    (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                     cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if min(width, height, frames) <= 0 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Streaming input needs known positive dimensions, frame count and FPS.")
        self.shape = (frames, height, width, 3)

    def iter_prepared(self, height, width, chunk_frames, tile, interrupt):
        from .vendor.runtime.inference.preprocessing.input import pil_center_crop_resize
        capture = cv2.VideoCapture(str(self.path))
        window, limit = [], chunk_frames + 1
        try:
            while True:
                interrupt()
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize one frame at a time, then retain only this spatial tile.
                resized = pil_center_crop_resize(rgb[None], height, width,
                                                 resample=Image.Resampling.BICUBIC)[0]
                y, x, h, w = tile
                window.append(np.ascontiguousarray(resized[y:y+h, x:x+w]))
                if len(window) == limit:
                    yield self._tensor(window)
                    window, limit = [], chunk_frames
            if window:
                yield self._tensor(window)
        finally:
            capture.release()

    @staticmethod
    def _tensor(frames):
        return (torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
                .float().div_(127.5).sub_(1))


class RGBTileWriter:
    def __init__(self, path, height, width, interrupt):
        self.path, self.height, self.width = Path(path), height, width
        self.interrupt = interrupt
        self.frame_count = 0
        self.file = self.path.open("wb")

    def append(self, video):
        if tuple(video.shape[:2]) != (1, 3) or tuple(video.shape[-2:]) != (self.height, self.width):
            raise ValueError("Unexpected streaming decode segment shape.")
        if shutil.disk_usage(self.path.parent).free < self.height * self.width * 3 * 4 + 128 * 2**20:
            raise RuntimeError("FlashVSR+ tile spool ran out of disk space. Choose a larger temp disk.")
        for index in range(video.shape[2]):
            self.interrupt()
            rgb = (video[0, :, index].detach().to(device="cpu", dtype=torch.float32)
                   .add(1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
                   .permute(1, 2, 0).contiguous().numpy())
            self.file.write(rgb.tobytes())
            self.frame_count += 1

    def close(self):
        self.file.close()


def mux_audio(video, source, destination):
    # Do not use -shortest: a short audio track must not truncate video frames.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-i", str(video),
               "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
               "-c:a", "copy", "-movflags", "+faststart", str(destination)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        audio_codec = command.index("-c:a") + 1
        command[audio_codec] = "aac"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("FlashVSR+ audio mux failed: " + result.stderr.decode(errors="replace")[-2000:])


class VideoFileSink:
    def __init__(self, destination, work_directory, *, crf=18, preset="veryfast", keep_audio=True):
        self.destination = Path(destination)
        self.work_directory = Path(work_directory)
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self.crf, self.preset, self.keep_audio = crf, preset, keep_audio

    def render(self, pipe, source, height, width, tiles, overlap, fps, progress):
        from .pipeline import interrupt
        # Compositing/encoding holds frames, never the complete timeline.
        ensure_cpu_output_fits(8, height, width)
        raw_bytes = source.shape[0] * sum(h * w for _, _, h, w in tiles) * 3
        # Reserve room for the encoded file, muxed copy, and temporary input.
        required = raw_bytes + source.shape[0] * height * width * 3 // 2 + 2**30
        free = shutil.disk_usage(self.work_directory).free
        if required > free * .9:
            raise RuntimeError(
                f"FlashVSR+ streaming needs about {required / 2**30:.2f} GiB temporary disk space; "
                f"only {free / 2**30:.2f} GiB is free at {self.work_directory}. "
                "Reduce scale/target size or use a ComfyUI temp directory on a larger disk."
            )
        progress(f"Streaming VIDEO: {width}x{height}, {len(tiles)} spatial tiles; "
                 f"estimated temp disk {required / 2**30:.2f} GiB")
        with tempfile.TemporaryDirectory(prefix="flashvsr-tiles-", dir=self.work_directory) as work:
            work = Path(work)
            spools, frames, metadata = [], None, {}
            for index, tile in enumerate(tiles):
                interrupt()
                progress(f"Spatial tile {index+1}/{len(tiles)} (continuous temporal stream)")
                path = work / f"tile-{index}.rgb"
                writer = RGBTileWriter(path, tile[2], tile[3], interrupt)
                try:
                    _, metadata = pipe._run_tile(source, height, width, tile, fps, progress,
                                                 segment_writer=writer)
                finally:
                    writer.close()
                if frames is not None and writer.frame_count != frames:
                    raise RuntimeError("Spatial tiles produced inconsistent frame counts.")
                frames = writer.frame_count
                spools.append(path)
            silent = work / "encoded.mp4"
            self._encode(spools, tiles, frames, height, width, overlap, fps, silent, progress, interrupt)
            try:
                if self.keep_audio:
                    mux_audio(silent, source.path, self.destination)
                else:
                    shutil.copyfile(silent, self.destination)
                capture = cv2.VideoCapture(str(self.destination))
                try:
                    actual = tuple(int(capture.get(k)) for k in
                                   (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH))
                finally:
                    capture.release()
                if actual != (frames, height, width):
                    raise RuntimeError(f"Encoded VIDEO frame/dimension mismatch: {actual}")
            except BaseException:
                self.destination.unlink(missing_ok=True)
                raise
        metadata.update({"source_frames": frames, "output_frames": frames, "output_path": str(self.destination),
                         "output_storage": "video_file", "tile_spool": "RGB8", "temp_disk_estimate_gib": required / 2**30,
                         "audio_passthrough": self.keep_audio, "temporal_state": "continuous_per_spatial_tile"})
        return str(self.destination), metadata

    def _encode(self, spools, tiles, frames, height, width, overlap, fps, destination, progress, interrupt):
        weights = np.zeros((height, width), dtype=np.float32)
        masks = [blend_mask(tile, height, width, overlap).numpy() for tile in tiles]
        for tile, mask in zip(tiles, masks):
            y, x, h, w = tile
            weights[y:y+h, x:x+w] += mask
        if not np.all(weights > 0):
            raise RuntimeError("Spatial streaming tiles left uncovered pixels.")
        encoder = imageio_ffmpeg.write_frames(
            str(destination), (width, height), fps=fps, codec="libx264", quality=None,
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
            output_params=["-crf", str(self.crf), "-preset", self.preset, "-threads", "4"],
            ffmpeg_log_level="error")
        with ExitStack() as stack:
            streams = [stack.enter_context(path.open("rb")) for path in spools]
            canvas = np.empty((height, width, 3), dtype=np.float32)
            try:
                encoder.send(None)
                for frame_index in range(frames):
                    interrupt()
                    canvas.fill(0)
                    for stream, tile, mask in zip(streams, tiles, masks):
                        y, x, h, w = tile
                        data = stream.read(h * w * 3)
                        if len(data) != h * w * 3:
                            raise RuntimeError("Truncated tile spool.")
                        rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                        canvas[y:y+h, x:x+w] += rgb * mask[:, :, None]
                    np.divide(canvas, weights[:, :, None], out=canvas)
                    np.rint(canvas, out=canvas)
                    np.clip(canvas, 0, 255, out=canvas)
                    encoder.send(canvas.astype(np.uint8))
                    if frame_index % 16 == 0 or frame_index + 1 == frames:
                        progress(f"Encoding video frame {frame_index+1}/{frames}")
            finally:
                encoder.close()
