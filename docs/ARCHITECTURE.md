# Official flow and node design

Source reviewed: OpenImagingLab/FlashVSR_Plus at `a08fb184a84cc2157c4f271234555fcda7902c87`, retrieved 2026-10-15.

## 1. inference/run_stream.py

The CLI parses mode, output dimensions, seed, decoder/cache controls and temporal compression. It calls `stream_config_for_mode(mode, ...)`, `model_paths_for_mode(...)`, `build_inference_pipeline(config, paths)`, then `pipeline.infer(InferenceRequest(...))`.

Turbo enables Hybrid-B25 with replacement layers 1–5, 7–11, 13–17, 19–23 and 25–29. Original sparse layers are 0,6,12,18,24. Base/V1 use the same non-Hybrid sparse configuration with distinct checkpoints. All presets use one flow-matching step, sigma shift 5, BF16, TinyDecoder, and four latent frames per stream chunk. RY-DMD is training/distillation, already reflected in the released checkpoints.

## 2. flashvsr_plus Python surface

`flashvsr_plus/__init__.py` re-exports `InferenceConfig`, `InferenceRequest`, `ModelPaths`, `ModelBundle`, `build_inference_pipeline`, and `stream_config_for_mode` from the original `inference` package. The package itself is a stable import surface, not a second inference implementation.

`build_model_bundle` constructs the DiT with official attention/LQ projection settings, loads DiT and projection checkpoints with ≥95% key coverage, constructs TinyDecoder or Wan VAE, creates the official FlowMatchInferenceScheduler, and loads the fixed context tensor. Safe tensor loading and `torch.load(weights_only=True)` follow upstream.

## 3. pipeline.infer

For `decode_every=auto`, input is loaded as `[1,3,F,H,W]` in [-1,1], center-cropped/resized, and expanded to the fixed temporal grid. Compression 1/2 repeats post-first frames by 4/2 respectively. After decoding, repeated groups are merged and synthetic tail frames are removed. Optional official wavelet correction runs against the matching LQ reference.

For numeric decode intervals, `iter_lq_video` yields bounded CPU windows into `TemporalVideoStream`. This expands the temporal grid incrementally, buffers only the live source window, and releases consumed frames. The same `StreamRunner` performs projection, denoising, incremental decoding and per-segment correction. Source frame count is restored after padding/merge. When writing a file, upstream can mux source audio; this plugin instead returns ComfyUI AUDIO unchanged.

## 4. StreamRunner

The LQ projection consumes the unique first frame plus four look-ahead pixel frames per latent, retaining its convolutional stream cache. A CUDA generator seeded from `config.seed` generates four-latent noise chunks. `DiT.forward_stream` retains per-layer causal KV state and calls the official attention/layout code. The one-step scheduler computes the released denoising update. Latent chunks move to CPU pending decode.

Numeric decoding temporarily offloads DiT and its cache to CPU while TinyDecoder runs. TinyDecoder preserves its temporal memories between contiguous segments. The runner merges temporal groups, applies wavelet correction and accumulates CPU output. It clears KV/LQ/decoder caches even after failures. State is per run, so reusable weights do not mean shared history across different clips.

## 5. ComfyUI adapter

`FlashVSRPlusLoader` -> `FlashVSRPlusPipe` owns the official pipeline and weights. It builds the bundle on CPU to avoid an initial all-model CUDA spike. The selected attention kernel is independently checked on the real CUDA device, then inserted into the unchanged official configuration. No global `sys.path` or generic `models` / `inference` aliases are installed.

```python
config = stream_config_for_mode(
    mode, device="cpu", hardware_profile="generic", attn_kernel="sdpa"
)
config.attn_kernel = verified_cuda_backend
pipeline = build_inference_pipeline(config, paths)
config.device = str(cuda_device)
```

`FlashVSRPlusUpscale` accepts one IMAGE batch or native VIDEO, converts RGB8 chunks with upstream `pil_center_crop_resize(..., BICUBIC)`, and feeds `[1,3,F,H,W]` CPU windows to the official `TemporalVideoStream` and `StreamRunner`. This replaces the CLI's OpenCV input reader without rewriting model math. When decode_every is auto (or decoder=vae), it calls the public `pipeline.infer(InferenceRequest(lq_video=...))` instead.

```python
source = iter_prepared(images, height, width, decode_every * temporal_compression)
stream = TemporalVideoStream(source, temporal_compression, 4)
try:
    result = pipeline.runner.run(stream, None, progress=progress)
    video = result.video[:, :, :stream.source_frames]
finally:
    stream.close()
```

Output converts [-1,1] -> [0,1] and `[1,3,F,H,W]` -> `[F,H,W,3]` on CPU, with an assertion that source frame count is preserved. Native VIDEO is detected using the installed ComfyUI API. Audio objects and FPS pass through independently of model execution.

## 6. Two automatic execution policies

Capacity ≥23GiB and free memory ≥18GiB selects `official`; smaller/busy cards select `low_vram`. This allows nominal 24GB devices to qualify despite WDDM reservations. Auto reevaluates free memory after releasing Comfy-managed GPU models. If only SDPA exists, large targets use spatial tiles because upstream SDPA expands compact block masks into large token masks. Explicit profile/tiling controls take precedence over that policy; OOM recovery is separately controlled.

The low-memory wrapper runs complete official streams on overlapping spatial tiles. Tile boundaries align to 128 pixels. Each tile has independent official KV/decoder state covering the entire source clip. Positive ramps, a weight sum and normalized CPU accumulation avoid uncovered/black output edges. Global context outside a tile is unavailable, so spatial tiling is an approximation.

Block offload wraps `DiTBlock.forward_stream` itself: upstream calls that method directly, so ordinary nn.Module forward hooks would miss it. Only the executing block moves to CUDA. A per-instance `denoiser.to` adapter also intercepts upstream's incremental `_restore_dit` so it does not accidentally bring all blocks back to CUDA. Small embeddings/projection/head remain available. Decoder offload wraps its actual decode entry points. No class-global monkey patches are used.

VAE tiling passes pixel-derived latent tile sizes/strides into the official `VAEDecoder` adapter. TinyDecoder does not support those VAE arguments; it uses the official serial/incremental execution plus optional whole-pipeline spatial tiling. The VAE API path uses full-timeline latent accumulation, so TinyDecoder is preferred for long clips.

## 7. Recovery, concurrency and limitations

Each PIPE has a reentrant lock. A module-wide lock serializes plugin GPU work and process-global seed changes across multiple PIPEs. CUDA operations run under the chosen device context. There is no background service or implicit ComfyUI restart.

On CUDA OOM, the failed attempt leaves its exception scope before clearing live caches/weights and the allocator. The complete request restarts, with unchanged seed/output resolution and smaller spatial tiles, block offload, cache offload, serial decoding and a small FFN token chunk. At most two retries occur. Other exceptions propagate rather than being mistaken for OOM. Interrupted or failed requests release the bundle even when normal residency was requested.

Independent queue calls reuse weights but intentionally reset temporal history; persistent streaming sessions across calls are not implemented. IMAGE results and native VIDEO extraction still use host RAM proportional to clip length. Disk-backed long-video decoding/encoding can be added as separate file IO nodes without changing this pipeline. No Sparse_SageAttention path is advertised without a verified implementation of the exact upstream mask contract.
