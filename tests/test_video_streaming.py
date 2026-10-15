"""Actual video I/O + unmodified official timeline, without pretrained weights."""
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
import torch

from flashvsr_plus_plugin.memory import ensure_cpu_output_fits
from flashvsr_plus_plugin.nodes import FlashVSRPlusUpscale
from flashvsr_plus_plugin.pipeline import FlashVSRPlusPipe
from flashvsr_plus_plugin.tiling import tile_grid
from flashvsr_plus_plugin.video_streaming import VideoFileSink, VideoFileSource
from flashvsr_plus_plugin.vendor.runtime.inference import FlashVSRInferencePipeline, ModelBundle, stream_config_for_mode
from flashvsr_plus_plugin.vendor.runtime.inference.loading.model_bundle import FlowMatchInferenceScheduler
from test_streaming import FakeDiT, IdentityDecoder


def lossless_clip(path, frames):
    writer = imageio_ffmpeg.write_frames(str(path), (128, 128), fps=24, codec="ffv1",
                                        pix_fmt_out="bgr0", macro_block_size=1,
                                        ffmpeg_log_level="error")
    writer.send(None)
    try:
        for index in range(frames):
            writer.send(np.full((128, 128, 3), 30 + index * 5, dtype=np.uint8))
    finally:
        writer.close()


@pytest.mark.parametrize("compression", [1, 2, 4])
@pytest.mark.parametrize("frames", [1, 18, 33])
def test_streaming_video_round_trip_preserves_timeline_and_overlapping_tiles(tmp_path, monkeypatch, frames, compression):
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt", lambda: None)
    source_path = tmp_path / "input.mkv"
    lossless_clip(source_path, frames)
    source = VideoFileSource(source_path)
    config = stream_config_for_mode("turbo", device="cpu", hardware_profile="generic",
                                    temporal_compression=compression, decode_every=4)
    config.dtype = "fp32"
    bundle = ModelBundle(FakeDiT(), IdentityDecoder(), FlowMatchInferenceScheduler(), torch.zeros(1, 2))
    pipe = FlashVSRPlusPipe.__new__(FlashVSRPlusPipe)
    pipe.pipeline = FlashVSRInferencePipeline(config, bundle)
    path = tmp_path / "output.mp4"
    sink = VideoFileSink(path, tmp_path, crf=0)
    result, metadata = sink.render(pipe, source, 128, 384, tile_grid(128, 384, 256, 128),
                                   128, source.fps, lambda _: None)
    assert result == str(path)
    assert metadata["output_frames"] == frames
    assert metadata["temporal_state"] == "continuous_per_spatial_tile"
    assert metadata["input_buffer_peak_frames"] < 40
    capture = cv2.VideoCapture(result)
    decoded = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (128, 384, 3)
            decoded.append(float(frame.mean()))
    finally:
        capture.release()
    assert len(decoded) == frames
    np.testing.assert_allclose(decoded, 30 + np.arange(frames) * 5, atol=3)
    assert not list(tmp_path.glob("flashvsr-tiles-*"))


def test_image_ram_guard_reproduces_report_without_large_allocation(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=40 * 2**30))
    with pytest.raises(RuntimeError, match="254.71 GiB RAM"):
        ensure_cpu_output_fits(360, 10752, 5888)
    assert ensure_cpu_output_fits(4, 128, 128) == 4 * 128 * 128 * 3 * 4


def test_native_video_guard_runs_before_full_components_decode(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=40 * 2**30))
    def materialize():
        pytest.fail("The oversized source must not be decoded into a full IMAGE batch")
    video = SimpleNamespace(get_dimensions=lambda: (1472, 2688), get_frame_count=lambda: 360,
                            get_components=materialize)
    with pytest.raises(RuntimeError, match="Streaming"):
        FlashVSRPlusUpscale().upscale(None, video=video, scale=4)


def test_streaming_disk_guard_precedes_model_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=2**30))
    sink = VideoFileSink(tmp_path / "output.mp4", tmp_path)
    source = SimpleNamespace(shape=(360, 2688, 1472, 3))
    with pytest.raises(RuntimeError, match="temporary disk space"):
        sink.render(None, source, 10752, 5888, [(0, 0, 10752, 5888)], 0, 24, lambda _: None)
    assert not list(tmp_path.glob("flashvsr-tiles-*"))


def test_streaming_spools_are_removed_on_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt", lambda: None)
    class Interrupted(RuntimeError):
        pass
    def fail(*args, segment_writer, **kwargs):
        segment_writer.append(torch.zeros(1, 3, 2, 128, 128))
        raise Interrupted("cancelled")
    sink = VideoFileSink(tmp_path / "output.mp4", tmp_path)
    source = SimpleNamespace(shape=(2, 128, 128, 3))
    with pytest.raises(Interrupted):
        sink.render(SimpleNamespace(_run_tile=fail), source, 128, 128,
                    [(0, 0, 128, 128)], 0, 24, lambda _: None)
    assert not list(tmp_path.glob("flashvsr-tiles-*"))
    assert not (tmp_path / "output.mp4").exists()


def test_short_source_audio_does_not_truncate_video(tmp_path):
    import subprocess
    from flashvsr_plus_plugin.video_streaming import mux_audio
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    silent, audio, output = (tmp_path / name for name in ("silent.mp4", "audio.m4a", "muxed.mp4"))
    subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=128x128:r=24:d=1", "-c:v", "libx264", str(silent)], check=True)
    subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=0.1", "-c:a", "aac", str(audio)], check=True)
    mux_audio(silent, audio, output)
    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    finally:
        capture.release()
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(output),
                    "-map", "0:a:0", "-f", "null", "-"], check=True)
