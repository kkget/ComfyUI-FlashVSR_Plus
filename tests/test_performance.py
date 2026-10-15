"""Residency skips repeated weight transfers without changing stream math."""
from types import SimpleNamespace
import threading

import pytest
import torch

from flashvsr_plus_plugin.memory import keep_models_resident
from flashvsr_plus_plugin.profiles import choose_profile


def test_residency_restores_instance_methods_on_failure_and_does_not_patch_other_models():
    denoiser, decoder, other = (torch.nn.Linear(2, 2) for _ in range(3))
    calls = []
    original = denoiser.to
    # Simulate an existing per-instance offload adapter.
    def move(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    denoiser.to = move
    bundle = SimpleNamespace(denoiser=denoiser, decoder=SimpleNamespace(module=decoder))
    with pytest.raises(RuntimeError, match="interrupted"):
        with keep_models_resident(bundle):
            assert denoiser.to("cpu") is denoiser
            assert not calls
            denoiser.to(dtype=torch.float64)
            assert denoiser.weight.dtype == torch.float64
            assert len(calls) == 1
            assert "to" not in other.__dict__
            raise RuntimeError("interrupted")
    assert denoiser.to is move
    assert "to" not in decoder.__dict__
    denoiser.to("cpu")
    assert len(calls) == 2


@pytest.mark.parametrize("free,size,offload", [(20,768,"resident"), (12,512,"resident"),
                                               (8,512,"block"), (5,384,"block"), (3,256,"block")])
def test_balanced_policy_retreats_when_free_vram_drops(free, size, offload):
    profile = choose_profile(24, free, "triton_block_sparse", "balanced")
    assert profile.tiled and profile.tile_size == size
    assert profile.offload == offload
    assert choose_profile(24, free, "sdpa", "balanced").tile_size <= 384
    assert choose_profile(24, free, "triton_block_sparse", "low_vram").offload == "block"


@pytest.mark.cuda
def test_resident_stream_matches_stage_offload_and_eliminates_repeated_weight_moves(monkeypatch):
    import json
    from flashvsr_plus_plugin.pipeline import FlashVSRPlusPipe
    from test_cuda import miniature_pipeline
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.prepare_device", lambda _: None)
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt", lambda: None)
    pipe = FlashVSRPlusPipe.__new__(FlashVSRPlusPipe)
    pipe.lock = threading.RLock()
    pipe.device = torch.device("cuda:0")
    pipe.mode, pipe.decoder_kind = "turbo", "tiny"
    pipe.profile_request, pipe.offload_request = "official", "model"
    pipe.backend, pipe.closed = "triton_block_sparse", False
    pipe._block_adapter_installed = pipe._decoder_offload_installed = False
    pipe._active_offload = None
    pipe.pipeline = miniature_pipeline(pipe.backend)
    denoiser = pipe.pipeline.bundle.denoiser
    original = denoiser.to
    cpu_moves = []
    def track(*args, **kwargs):
        if args and str(args[0]) == "cpu":
            cpu_moves.append(1)
        return original(*args, **kwargs)
    denoiser.to = track
    images = torch.rand(18, 32, 48, 3)
    try:
        reference, _ = pipe.upscale(images, tiled="disabled", oom_retry=False)
        assert len(cpu_moves) > 1
        cpu_moves.clear()
        pipe.offload_request = "resident"
        actual, metadata = pipe.upscale(images, tiled="disabled", oom_retry=False)
        assert len(cpu_moves) == 1  # End-of-request cleanup only.
        assert denoiser.to is track
        assert next(denoiser.parameters()).device.type == "cpu"
        assert json.loads(metadata)["dit_resident_during_decode"] is True
        torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
        # Fail inside the resident context after CUDA weights have moved. The
        # CPU transfer methods must be restored before recovery unloads them.
        forward = denoiser.forward_stream
        calls = []
        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("resident execution failure")
            return forward(*args, **kwargs)
        denoiser.forward_stream = fail_once
        recovered, metadata = pipe.upscale(images, tiled="disabled", oom_retry=True)
        info = json.loads(metadata)
        assert info["offload"] == "block" and len(info["oom_retries"]) == 1
        assert info["dit_resident_during_decode"] is False
        assert next(denoiser.parameters()).device.type == "cpu"
        torch.testing.assert_close(recovered, reference, atol=.02, rtol=.02)
    finally:
        pipe.close()
