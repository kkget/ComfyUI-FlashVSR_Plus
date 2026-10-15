# Validation record

Date: 2026-10-15 (Asia/Shanghai).

## Tested environment

- Windows, Python 3.12.11
- PyTorch 2.10.0+cu130
- NVIDIA GeForce RTX 5090 Laptop GPU
- Reported total memory: 23.889 GiB; free memory at environment check: approximately 22.55 GiB
- Installed ComfyUI Desktop core provides `comfy_api.latest` VIDEO components
- `block_sparse_attn` and external `flashvsr_attn` are not installed
- Triton is already available; the plugin's bundled official Triton operator passes an actual CUDA call

The automatic result on this machine is `official`, `triton_block_sparse`, `tiled=false`, `offload=model`, `cache_offload=auto`, `decode_every=4`.

## Automated checks

**132 tests passed**, including the optional real CUDA tests:

```bash
python -m pytest -q tests --run-cuda --rootdir=tests --confcutdir=tests --basetemp=/path/to/writable/fresh-temp --maxfail=1 --tb=short
```

The explicit basetemp avoids an unrelated access restriction on this machine's pre-existing system pytest temp directory. Use a fresh writable temporary path on your own machine. `--rootdir=tests --confcutdir=tests` avoids pytest trying to import a custom-node folder containing a hyphen as a normal Python package.

Coverage includes:

- Nominal 24GB recognition, busy GPUs and smaller GPUs selecting the expected profiles; explicit overrides and SDPA tile sizing.
- Native VIDEO-like input contract, AUDIO object and fractional source FPS preservation.
- Explicit output-size rejection and scale alignment to 128.
- Full output coverage and normalized overlapping tile reconstruction, including corners and 128-pixel dimensions.
- Bounded IMAGE reader windows, matching the CLI's first-window look-ahead and RGB8 bicubic preprocessing.
- Correct V1 source names and file formats, manual missing-model behavior, mocked download mapping and atomic file copy reuse.
- The unmodified official pipeline/StreamRunner at compression 1/2/4, frame counts 1/2/3/4/5/16/17/18/33, and decode intervals auto/4/8. An identity decoder makes frame loss, duplication or ordering changes directly observable. Each configuration is reused for a second clip to check state reset.
- The plugin's own incremental IMAGE-to-TemporalVideoStream adapter over the same tail-frame cases.
- **Real CUDA numerical comparison:** bundled official Triton block-sparse attention against dense SDPA with the same expanded block mask, BF16, tolerance atol/rtol 0.02.
- **Real official DiT + TinyDecoder execution:** reduced channel count/layer count with synthetic weights, Hybrid and original sparse layers, SDPA and Triton paths, eighteen source frames, normal execution versus block/decoder offload. These retain upstream implementations and intentionally reduce test memory/time.
- **Real PIPE adapter execution:** reduced official model, IMAGE preprocessing, repeated requests and a deliberately injected CUDA OOM followed by full request restart, smaller tiles and stronger offload, then another successful call and explicit unload.

The only warning was the environment's pre-existing `pynvml` deprecation warning from Torch.

## Additional checks against installed ComfyUI

- All five node classes import and expose their schemas using the actual hyphenated plugin module name.
- Native `VideoFromComponents` round-trip preserves the original IMAGE tensor, AUDIO object and source FPS.
- All three UI JSON workflows have valid link indices/types and match the installed plugin's input names.
- Both API JSON examples provide all required plugin inputs.
- No global `models`, `inference` or `model_io` package aliases are introduced.
- Vendored Python file hashes and the bundled official context tensor match `vendor/manifest.json`.
- Python compilation of plugin/source files succeeds.

Example commands:

```bash
python scripts/check_install.py --comfyui-path /path/to/ComfyUI/core
```

Use the Python interpreter belonging to ComfyUI. `--comfyui-path` points to the source directory containing `comfy/` and `comfy_api/`, which can differ from the models/custom_nodes user directory in Desktop installations.

## Limits of this validation

Complete DiT/projection/decoder weights were **not downloaded or run** during this implementation. No full pretrained restoration result, image-quality equivalence, official-preset 1080p/2K peak-memory figure, or speed benchmark is claimed. Wan VAE tiled decode is connected to the upstream API but was not executed with its full pretrained checkpoint. Real BSA execution was unavailable on this machine; selection falls through to the tested bundled Triton backend.

The automatic memory policies and OOM restart were tested functionally; the smaller-GPU policies were exercised using controlled capacity values and a synthetic OOM, not physical 8/12/16GB cards. Tests of reduced models verify execution/numerics/state handling, not pretrained quality or full-model peak VRAM.

Linux and ComfyUI versions without native VIDEO were not run here. The code feature-detects that API and keeps IMAGE/AUDIO registration available. The JSON files were structurally checked against the installed APIs; no live ComfyUI queue run or screenshot of the browser canvas is claimed.

Suggested first pretrained check: import `native_video.json`, choose a short source clip, use turbo/auto/tiny at a small 128-aligned target, then increase to 1920×1024 or 1920×1152 and inspect the returned JSON for backend, tile count, retry history and actual peak allocated memory. Compare `tiled=disabled` and `enabled` on the same clip/seed before choosing a long-video policy.
