"""Short full-weight transfer benchmark; no downloads or running ComfyUI needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.size < 128 or args.size % 128 or not 2 <= args.frames <= 65:
        parser.error("size must be a positive multiple of 128; frames must be 2..65")
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("flashvsr_plus_plugin")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    import torch
    from flashvsr_plus_plugin.models import ensure_models
    from flashvsr_plus_plugin import pipeline as adapter
    # This separate process owns no Comfy-managed models and no server queue.
    adapter.prepare_device = lambda device: torch.cuda.set_device(device)
    adapter.interrupt = lambda: None
    paths = ensure_models(Path(args.model_directory), "turbo", "tiny", False)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    pipe = adapter.FlashVSRPlusPipe(mode="turbo", paths=paths, device=device,
                                   profile="balanced", offload="model", attn_backend="auto")
    loading = time.perf_counter() - started
    images = torch.rand(args.frames, args.size, args.size, 3,
                        generator=torch.Generator().manual_seed(42))
    report = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "backend": pipe.backend, "shape": list(images.shape), "scale": 1.0,
              "model_load_seconds": loading, "warmup": "one stage-offload run; excluded",
              "scope": "synthetic IMAGE, full Turbo weights, one tile, not full-video throughput",
              "runs": []}
    def run(mode):
        pipe.offload_request = mode
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output, metadata = pipe.upscale(images, scale=1, tiled="disabled", oom_retry=False)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        info = json.loads(metadata)
        return output, {"offload": mode, "seconds_including_cleanup": seconds,
                        "peak_allocated_gib": info["peak_allocated_gib"],
                        "tiny_decoder_execution_mode": info["tiny_decoder_execution_mode"],
                        "oom_retries": info["oom_retries"]}
    try:
        print(f"Warmup: {args.frames} frames at {args.size}x{args.size}", flush=True)
        reference, _ = run("model")
        # Block adapters are intentionally sticky, so that comparison runs last.
        for mode in ("model", "resident", "block"):
            print(f"Benchmark: offload={mode}", flush=True)
            actual, result = run(mode)
            difference = (actual - reference).abs()
            result.update({"max_absolute_difference": difference.max().item(),
                           "mean_absolute_difference": difference.mean().item()})
            torch.testing.assert_close(actual, reference, atol=.02, rtol=.02)
            report["runs"].append(result)
            del actual, difference
            print(json.dumps(result), flush=True)
    finally:
        pipe.close()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
