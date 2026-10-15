"""Real official attention/DiT/TinyDecoder integration at reduced channel counts.

No pretrained model downloads. This verifies execution, not restoration quality.
"""
from pathlib import Path
import threading

import pytest
import torch

from flashvsr_plus_plugin.memory import install_block_offload, install_decoder_offload, release_bundle
from flashvsr_plus_plugin.profiles import probe_backend
from flashvsr_plus_plugin.vendor.runtime.inference import (
    ComponentFactories, InferenceRequest, ModelPaths, build_inference_pipeline, stream_config_for_mode,
)
from flashvsr_plus_plugin.vendor.runtime.models import FlashVSRModelConfig, build_model
from flashvsr_plus_plugin.vendor.runtime.models.config import DiTConfig, BlockReplacementConfig, LQProjectionConfig
from flashvsr_plus_plugin.vendor.runtime.inference.decoders.tiny_model import TAEHV


@pytest.mark.cuda
def test_bundled_triton_matches_dense_mask_reference():
    from flashvsr_plus_plugin.vendor.runtime.flashvsr_attn import stream_block_sparse_attention
    device = torch.device("cuda:0")
    probe_backend("triton_block_sparse",device)
    with torch.inference_mode():
        torch.manual_seed(42)
        q = torch.randn(1,2,2,128,128,device=device,dtype=torch.bfloat16)
        k = torch.randn(1,2,3,128,128,device=device,dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.tensor([[[[1,1,0],[0,1,1]]]],device=device,dtype=torch.bool).expand(1,2,2,3).contiguous()
        actual = stream_block_sparse_attention(q,k,v,mask)
        token_mask = mask.repeat_interleave(128,-2).repeat_interleave(128,-1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.flatten(2,3),k.flatten(2,3),v.flatten(2,3),attn_mask=token_mask)
        torch.testing.assert_close(actual.flatten(2,3),expected,atol=.02,rtol=.02)


def miniature_pipeline(backend):
    config = stream_config_for_mode("turbo",device="cpu",hardware_profile="generic",
                                    decode_every=4,tiny_decoder_mode="serial")
    config.attn_kernel = backend
    def denoiser(conf,path):
        small = FlashVSRModelConfig(dit=DiTConfig(dim=128,num_heads=1,ffn_dim=256,num_layers=2),
                                   lq_projection=LQProjectionConfig(num_injection_layers=1),
                                   attention=conf.attention_config(),replacement=BlockReplacementConfig(layers=(1,)))
        return build_model(small)
    factories = ComponentFactories(denoiser=denoiser,
        tiny_decoder=lambda conf,path: TAEHV(checkpoint_path=None,channels=[8,8,8,8],latent_channels=784))
    prompt = Path(__file__).resolve().parents[1] / "assets" / "posi_prompt.pth"
    pipeline = build_inference_pipeline(config,ModelPaths(context=str(prompt)),factories=factories)
    config.device = "cuda:0"
    return pipeline


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sdpa","triton_block_sparse"])
def test_actual_official_dit_decoder_and_block_offload(backend):
    torch.manual_seed(42)
    pipeline = miniature_pipeline(backend)
    bundle = pipeline.bundle
    request = InferenceRequest(lq_video=torch.rand(1,3,18,128,128).mul(2).sub(1),fps=24)
    with torch.inference_mode():
        bundle.denoiser.to("cuda:0")
        bundle.decoder.module.to("cuda:0")
        reference = pipeline.infer(request).video
        release_bundle(bundle)
        install_block_offload(bundle.denoiser)
        install_decoder_offload(bundle.decoder,torch.device("cuda:0"),is_vae=False)
        bundle.denoiser.to("cuda:0")
        assert all(next(block.parameters()).device.type == "cpu" for block in bundle.denoiser.blocks)
        actual = pipeline.infer(request).video
        assert tuple(actual.shape) == (1,3,18,128,128)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,reference,rtol=.02,atol=.02)
        assert all(next(block.parameters()).device.type == "cpu" for block in bundle.denoiser.blocks)
        assert next(bundle.decoder.module.parameters()).device.type == "cpu"
        release_bundle(bundle)


@pytest.mark.cuda
def test_complete_pipe_reuse_and_oom_restart(monkeypatch):
    import json
    from flashvsr_plus_plugin.pipeline import FlashVSRPlusPipe
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.prepare_device",lambda device: None)
    monkeypatch.setattr("flashvsr_plus_plugin.pipeline.interrupt",lambda: None)
    pipe = FlashVSRPlusPipe.__new__(FlashVSRPlusPipe)
    pipe.lock = threading.RLock()
    pipe.device = torch.device("cuda:0")
    pipe.mode,pipe.decoder_kind = "turbo","tiny"
    pipe.profile_request,pipe.offload_request = "official","model"
    pipe.backend,pipe.closed = "triton_block_sparse",False
    pipe._block_adapter_installed = pipe._decoder_offload_installed = False
    pipe._active_offload = None
    pipe.pipeline = miniature_pipeline(pipe.backend)
    images = torch.rand(3,32,48,3)
    reference,_ = pipe.upscale(images,scale=1,tiled="disabled",oom_retry=False)
    original = pipe._run_once
    calls = []
    def fail_once(*args,**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("deliberate recovery test")
        return original(*args,**kwargs)
    pipe._run_once = fail_once
    actual,info = pipe.upscale(images,scale=1,tiled="disabled",oom_retry=True)
    metadata = json.loads(info)
    assert len(calls) == 2 and len(metadata["oom_retries"]) == 1
    assert metadata["offload"] == "block"
    assert metadata["tiled"] is True
    torch.testing.assert_close(actual,reference,atol=.02,rtol=.02)
    # A following clip still works after recovery, with a clean stream history.
    third,_ = pipe.upscale(images,scale=1,tiled="disabled",oom_retry=False)
    torch.testing.assert_close(third,reference,atol=.02,rtol=.02)
    pipe.close()
    assert pipe.closed and pipe.pipeline is None


@pytest.mark.cuda
def test_actual_cuda_pipe_streams_to_file_and_recovers_oom(tmp_path, monkeypatch):
    import json
    from flashvsr_plus_plugin.pipeline import FlashVSRPlusPipe
    from test_video_streaming import lossless_clip
    from flashvsr_plus_plugin.video_streaming import VideoFileSource, VideoFileSink
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
    path = tmp_path / "source.mkv"
    lossless_clip(path, 18)
    source = VideoFileSource(path)
    sink = VideoFileSink(tmp_path / "output.mp4", tmp_path)
    original = pipe._run_once
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("deliberate file-output recovery test")
        return original(*args, **kwargs)
    pipe._run_once = fail_once
    try:
        result, info = pipe.upscale(source, scale=1, tiled="disabled", _video_sink=sink)
        metadata = json.loads(info)
        assert Path(result).is_file()
        assert metadata["output_frames"] == 18
        assert metadata["output_storage"] == "video_file"
        assert len(calls) == 2 and len(metadata["oom_retries"]) == 1
        assert metadata["offload"] == "block"
        assert not list(tmp_path.glob("flashvsr-tiles-*"))
    finally:
        pipe.close()
