"""Check the plugin's IMAGE source adapter through the real official stream runner."""
import threading

import pytest
import torch

from flashvsr_plus_plugin.pipeline import FlashVSRPlusPipe
from flashvsr_plus_plugin.frames import iter_prepared
from flashvsr_plus_plugin.vendor.runtime.inference import (
    FlashVSRInferencePipeline, ModelBundle, stream_config_for_mode,
)
from flashvsr_plus_plugin.vendor.runtime.inference.loading.model_bundle import FlowMatchInferenceScheduler
from test_streaming import FakeDiT, IdentityDecoder


@pytest.mark.parametrize("frames", [1,2,3,4,5,16,17,18,33])
@pytest.mark.parametrize("compression", [1,2,4])
def test_image_reader_matches_official_timeline(monkeypatch,frames,compression):
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt",lambda: None)
    config = stream_config_for_mode("turbo",device="cpu",hardware_profile="generic",
                                    temporal_compression=compression,decode_every=4)
    config.dtype = "fp32"
    bundle = ModelBundle(FakeDiT(),IdentityDecoder(),FlowMatchInferenceScheduler(),torch.zeros(1,2))
    pipe = FlashVSRPlusPipe.__new__(FlashVSRPlusPipe)
    pipe.pipeline = FlashVSRInferencePipeline(config,bundle)
    images = torch.linspace(.05,.95,frames).view(frames,1,1,1).expand(frames,32,48,3).clone()
    video,metadata = pipe._run_tile(images,128,256,(0,128,128,128),24,lambda message: None)
    expected = torch.cat(list(iter_prepared(images,128,256,16,(0,128,128,128))),dim=2)
    torch.testing.assert_close(video,expected)
    assert metadata["output_frames"] == frames


def test_cpu_tile_accumulation_has_no_black_edges(monkeypatch):
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt",lambda: None)
    from flashvsr_plus_plugin.tiling import tile_grid
    pipe = FlashVSRPlusPipe.__new__(FlashVSRPlusPipe)
    pipe.device = "cpu"
    pipe._active_offload = "disabled"
    pipe.pipeline = type("Pipeline",(),{"bundle": type("Bundle",(),{
        "denoiser": torch.nn.Linear(1,1),
        "decoder": type("Decoder",(),{"module": torch.nn.Linear(1,1)})()})()})()
    def fake_tile(images,height,width,tile,fps,progress):
        return torch.full((1,3,images.shape[0],tile[2],tile[3]),-.25),{}
    pipe._run_tile = fake_tile
    images = torch.zeros(2,32,32,3)
    result,_ = pipe._run_once(images,384,640,tile_grid(384,640,256,128),128,24,lambda msg: None)
    torch.testing.assert_close(result,torch.full_like(result,.375))
