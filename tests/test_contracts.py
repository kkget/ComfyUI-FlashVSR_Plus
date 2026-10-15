from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from flashvsr_plus_plugin.frames import iter_prepared, target_size, unpack_input
from flashvsr_plus_plugin.models import ensure_models, model_files
from flashvsr_plus_plugin.profiles import choose_profile
from flashvsr_plus_plugin.tiling import blend_mask, tile_grid


@pytest.mark.parametrize("total,free,expected", [(23.89,22,"official"), (24,12,"low_vram"),
                                               (16,15,"low_vram"), (32,27,"official")])
def test_auto_vram_policy(total, free, expected):
    profile = choose_profile(total, free, "triton_block_sparse")
    assert profile.name == expected
    assert profile.tiled == (expected == "low_vram")


def test_manual_profile_and_sdpa_budget():
    assert choose_profile(16,12,"sdpa","official").name == "official"
    assert choose_profile(16,12,"sdpa").tile_size <= 384


def test_video_audio_and_fps_contract():
    images = torch.rand(3, 16, 16, 3)
    audio = {"waveform": torch.zeros(1,2,480), "sample_rate": 48000}
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(
        images=images, audio=audio, frame_rate=Fraction(30000,1001)))
    out, sound, fps = unpack_input(video=video)
    assert out is images and sound is audio
    assert fps == float(Fraction(30000,1001))
    with pytest.raises(ValueError, match="exactly one"):
        unpack_input(images=images, video=video)


def test_size_alignment_and_no_silent_explicit_rounding():
    images = torch.zeros(1,270,480,3)
    assert target_size(images,4,0,0) == (1152,1920)
    with pytest.raises(ValueError):
        target_size(images,4,1080,1920)
    with pytest.raises(ValueError):
        target_size(images,4,0,1920)


@pytest.mark.parametrize("height,width,size,overlap", [(1152,1920,512,128), (128,128,256,128),
                                                     (768,1024,384,128), (512,512,256,0)])
def test_blending_covers_boundaries_and_reconstructs_constant(height,width,size,overlap):
    total = torch.zeros(height,width)
    values = torch.zeros_like(total)
    for tile in tile_grid(height,width,size,overlap):
        y,x,h,w = tile
        assert all(value % 128 == 0 for value in tile)
        mask = blend_mask(tile,height,width,overlap)
        total[y:y+h,x:x+w] += mask
        values[y:y+h,x:x+w] += mask * .375
    assert (total > 0).all()
    torch.testing.assert_close(values / total, torch.full_like(total,.375))


@pytest.mark.parametrize("size,overlap", [(128,0),(512,512),(512,64),(300,128)])
def test_reject_bad_tiles(size,overlap):
    with pytest.raises(ValueError):
        tile_grid(1024,1024,size,overlap)


def test_streamed_preprocessing_uses_bounded_windows():
    images = torch.rand(38,16,24,3)
    chunks = list(iter_prepared(images,128,256,16,(0,128,128,128)))
    assert [x.shape[2] for x in chunks] == [17,16,5]
    assert all(tuple(x.shape[:2]) == (1,3) for x in chunks)
    assert all(x.shape[-2:] == (128,128) for x in chunks)


def test_v1_ckpt_keeps_real_format():
    files = model_files("v1")
    assert files[0].remote == "diffusion_pytorch_model_streaming_dmd.safetensors"
    assert files[1].local.endswith(".ckpt") and files[1].remote.endswith(".ckpt")


def test_missing_manual_model_does_not_download(tmp_path):
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")):
        with pytest.raises(FileNotFoundError, match="auto_download"):
            ensure_models(tmp_path,"turbo","tiny",False)


def test_download_mapping_is_atomic_and_reusable(tmp_path):
    downloaded = tmp_path / "fake-download"
    downloaded.write_bytes(b"fake weight bytes; not used for inference")
    root = tmp_path / "models"
    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)) as fetch:
        result = ensure_models(root,"v1","tiny",True)
        assert fetch.call_count == 3
        assert Path(result.lq_projection).suffix == ".ckpt"
        ensure_models(root,"v1","tiny",False)
        assert fetch.call_count == 3
    assert not list(root.rglob("*.tmp"))
    assert (root / "prompt_tensor" / "posi_prompt.pth").is_file()
