"""Exercise the unmodified upstream runner with a controlled identity decoder.

These tests validate timeline/frame preservation, not model quality.
"""
from types import SimpleNamespace

import pytest
import torch

from flashvsr_plus_plugin.vendor.runtime.inference import (
    FlashVSRInferencePipeline, InferenceRequest, ModelBundle, stream_config_for_mode,
)
from flashvsr_plus_plugin.vendor.runtime.inference.loading.model_bundle import FlowMatchInferenceScheduler


class FakeProjection(torch.nn.Module):
    def clear_cache(self):
        pass

    def stream_forward(self, value):
        return [torch.zeros(1,1,1)]


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.LQ_proj_in = FakeProjection()
        self.blocks = []

    def forward_stream(self, value, *args, **kwargs):
        return torch.zeros_like(value)


class IdentityDecoder:
    def __init__(self):
        self.module = torch.nn.Linear(1,1)

    def reset(self):
        pass

    def decode(self, latents, *, lq_video, **kwargs):
        return SimpleNamespace(video=lq_video.clone())

    def decode_stream_segment(self, latents, lq_video, **kwargs):
        return SimpleNamespace(video=lq_video.clone())


@pytest.mark.parametrize("compression", [1,2,4])
@pytest.mark.parametrize("frames", [1,2,3,4,5,16,17,18,33])
@pytest.mark.parametrize("interval", ["auto",4,8])
def test_official_stream_preserves_every_source_frame(compression,frames,interval):
    config = stream_config_for_mode("turbo",device="cpu",hardware_profile="generic",
                                    temporal_compression=compression,decode_every=interval)
    config.dtype = "fp32"
    bundle = ModelBundle(FakeDiT(),IdentityDecoder(),FlowMatchInferenceScheduler(),torch.zeros(1,2))
    pipeline = FlashVSRInferencePipeline(config,bundle)
    # Different colors at each frame detect frame loss, duplication, or reordering.
    source = torch.linspace(-.9,.9,frames).view(1,1,frames,1,1).expand(1,3,frames,128,128).clone()
    result = pipeline.infer(InferenceRequest(lq_video=source,fps=29.97))
    assert result.frames == frames
    assert result.fps == 29.97
    torch.testing.assert_close(result.video,source)
    # Reusing the same PIPE must start a fresh temporal history for a new clip.
    second = pipeline.infer(InferenceRequest(lq_video=source.flip(2),fps=29.97))
    torch.testing.assert_close(second.video,source.flip(2))
