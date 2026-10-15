"""Read-only environment check and tiny real CUDA backend probe; no model download."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-path", default="")
    parser.add_argument("--backend", default="auto", choices=["auto","sdpa","triton_block_sparse","block_sparse"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("flashvsr_plus_plugin")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    if args.comfyui_path:
        sys.path.insert(0,str(Path(args.comfyui_path).resolve()))
    import torch
    from flashvsr_plus_plugin.profiles import choose_profile, resolve_backend
    from flashvsr_plus_plugin.frames import native_video_available
    result = {"python": sys.version, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "native_video": native_video_available()}
    for name in ("cv2","einops","safetensors","huggingface_hub","imageio_ffmpeg"):
        importlib.import_module(name)
    if result["cuda"]:
        device = torch.device("cuda",torch.cuda.current_device())
        backend = resolve_backend(args.backend,device)
        free,total = torch.cuda.mem_get_info(device)
        result.update({"gpu": torch.cuda.get_device_name(device),
                       **choose_profile(total/2**30,free/2**30,backend).to_dict()})
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["cuda"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
