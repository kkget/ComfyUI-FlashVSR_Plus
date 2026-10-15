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

The original implementation passed **132 tests**, including the optional real CUDA tests:

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

After fixing Desktop multi-directory model lookup, the updated default suite passed **137 tests, with 4 optional CUDA tests skipped**. New regression checks cover an empty shared folder before a populated local folder, automatic downloads enabled and disabled, Turbo/Base/V1, shared decoder reuse, V1 aliases, preserving configured path priority and missing-file path reporting. Hub downloads are blocked in the local-model tests so any unexpected network attempt fails the test. The optional CUDA tests were not rerun for this model-path change.

The installed Desktop extra-model-path configuration was also loaded in a separate diagnostic process. Its first shared directory had no weights; the second local directory contained all three Turbo/tiny weights. With `auto_download=true` and Hub downloading blocked, the fixed resolver selected all three local files and made **zero Hub calls**. This diagnostic did not start, restart or submit a generation to ComfyUI.

The subsequent file-backed VIDEO implementation passed **156 tests including all 5 CUDA tests**. New coverage includes real FFmpeg/OpenCV file round-trips through the unchanged official runner at temporal compression 1/2/4, 1/18/33 source frames, overlapping spatial tiles, bounded input windows, disk-space rejection, interruption cleanup, rejection of the reported 254.71 GiB IMAGE allocation before full native VIDEO decoding, and retaining all 24 video frames when source audio lasts only 0.1 seconds. A real CUDA reduced-model test exercises file output plus a deliberately injected OOM/restart and block offload.

## Additional checks against installed ComfyUI

- All six current node classes import and expose their schemas against the installed core; the original five were also checked using the actual hyphenated plugin module name.
- Native `VideoFromComponents` round-trip preserves the original IMAGE tensor, AUDIO object and source FPS.
- All five UI JSON workflows have valid link indices/types and matching named/positional plugin widget values, including the optional balanced Streaming example.
- All four API JSON examples provide all required plugin inputs.
- No global `models`, `inference` or `model_io` package aliases are introduced.
- Vendored Python file hashes and the bundled official context tensor match `vendor/manifest.json`.
- Python compilation of plugin/source files succeeds.

Example commands:

```bash
python scripts/check_install.py --comfyui-path /path/to/ComfyUI/core
```

Use the Python interpreter belonging to ComfyUI. `--comfyui-path` points to the source directory containing `comfy/` and `comfy_api/`, which can differ from the models/custom_nodes user directory in Desktop installations.

## Speed and residency checks

The speed update adds CPU coverage for balanced-policy thresholds, explicit low-VRAM overrides, SDPA tile bounds, per-instance transfer-method restoration on exceptions, and absence of class/global patches. A real CUDA reduced-model test compares model-stage offload with resident weights, verifies CPU transfers occur only at request cleanup, and injects an OOM inside resident execution to check method restoration, block-offload restart, output equivalence and CPU cleanup. Existing official timeline/file-output tests remain applicable.

Final validation of this update: **163 tests passed, including all 6 real CUDA tests**. Five UI graphs and four API examples were separately checked for input/link consistency, matching positional/named widgets and the retained 1x default. No live ComfyUI queue or complete user workflow was executed.

`scripts/benchmark_speed.py` ran against the **complete official Turbo DiT, LQ projection and TinyDecoder weights**, local-only, on this RTX 5090 Laptop. Synthetic seeded RGB float32 IMAGE input, BF16 execution, bundled Triton, temporal compression 4, decode interval 4, one untiled stream and the same TinyDecoder auto setting were used for each comparison. One model-stage-offload warmup was excluded; each measured time includes request cleanup but excludes model loading and file encoding. Each setting has one measured sample, not a statistical benchmark.

| Input | Model-stage offload | Resident weights | Peak allocated, model / resident | Output max absolute difference |
|---|---|---|---|---|
| 17 frames, 512×512 | 3.6065s | 2.6968s | 3.8250 / 6.9861GiB | 0 |
| 17 frames, 768×768 | 6.5213s | 4.6754s | 8.4368 / 11.6163GiB | 0 |

Separate block-offload runs with the same auto decoder took 3.0017s and 5.1573s respectively, with max output difference 0. These isolate weight transfers; the older low-VRAM profile also forces serial decoder/KV offload and is not identical to this block-only comparison. Residency improves throughput over model-stage offload by about 34% and 39% in these two samples, at a higher VRAM cost. Changing spatial tile size is a separate quality/performance tradeoff and was not claimed to preserve output elements.

A separate **17-frame synthetic IMAGE input, 1536×2688 output** ran successfully with full Turbo weights, `official`, `offload=model`, untiled, decoder auto (selected serial), KV auto and no OOM retries. It returned exactly `[17,2688,1536,3]`, took **31.8899s including cleanup** and peaked at **12.8780GiB PyTorch allocated**. This first run at that shape had no matching-shape warmup and excluded model loading and file codecs; it is not directly comparable with the warmed square-tile tests, nor a measurement of 360-frame execution. No 360-frame/long VIDEO throughput benchmark was run.

## Limits of this validation

The complete Turbo DiT, LQ projection and TinyDecoder checkpoints have been downloaded and verified against their published SHA256 hashes. Local path resolution was verified without Hub access. Initial validation ran a **single synthetic 128x128 frame** through the complete official Turbo weights and both actual Loader/Streaming VIDEO node methods, using `low_vram`, block offload and bundled Triton. Native `VideoFromFile` input was normalized to temporary Matroska and the encoded output verified as exactly one 128x128 frame. The PIPE was explicitly closed afterward. Subsequent short full-weight IMAGE timing/residency checks are documented above; no restoration-quality equivalence or full-length VIDEO speed/memory guarantee is claimed. Wan VAE tiled decode was not executed with its full pretrained checkpoint. Real BSA execution was unavailable; bundled official Triton was used.

The automatic memory policies and OOM restart were tested functionally; the smaller-GPU policies were exercised using controlled capacity values and a synthetic OOM, not physical 8/12/16GB cards. Tests of reduced models verify execution/numerics/state handling, not pretrained quality or full-model peak VRAM.

Linux and ComfyUI versions without native VIDEO were not run here. The code feature-detects that API and keeps IMAGE/AUDIO registration available. The JSON files were structurally checked against the installed APIs; no live ComfyUI queue run or screenshot of the browser canvas is claimed.

Suggested first full-resolution check: import `streaming_video.json`, choose a short source clip, use turbo/auto/tiny with the new `scale=1.0`, `height=width=0` defaults (or 0.5x for a large input), then increase dimensions while checking temporary disk requirements and the returned backend/tile/retry metadata. The IMAGE and Streaming nodes, Python PIPE and all UI/API examples use 1x as their starting default; this does not establish a full-resolution memory guarantee. Use `native_video.json` when IMAGE output is required and RAM is sufficient. Compare tiled and untiled output on the same short clip/seed before choosing a long-video policy.
