# ComfyUI-FlashVSR_Plus

官方 [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus) 的 ComfyUI 节点：Turbo / Base / V1、可复用 PIPE、IMAGE / 原生 VIDEO 输入、自动显存策略、完整时序处理和可选音频透传。

社区适配插件，使用官方代码与权重；并非 OpenImagingLab 官方发布的 ComfyUI 扩展。模型权重单独下载。

[倍率与内存](#scale-and-memory) · [中文快速开始](#中文快速开始) · [安装](#中文安装) · [模型下载与目录](#模型下载与目录) · [工作流](#示例工作流) · [故障排查](#常见问题) · [English](#english-installation-and-usage) · [实现分析](docs/ARCHITECTURE.md) · [验证记录](docs/VALIDATION.md)

<a id="scale-and-memory"></a>

## 先看倍率：默认 1×，从低内存开始

**所有示例工作流、普通 IMAGE / VIDEO 节点和 Streaming VIDEO 节点均默认 `scale=1.0`、`height=0`、`width=0`。先用短视频验证，再根据资源增加倍率。** 模型预设 Turbo / Base / V1 不决定输出倍率；自动显存策略也不会自动降低你指定的输出尺寸。

**Start at `scale=1.0`, `height=width=0` with a short clip.** All examples and both upscale nodes now use this default. Scale multiplies **both** dimensions: 2× means approximately 4× pixels; 4× means approximately 16× pixels. Increase it only after checking RAM, VRAM and temporary disk space. See the [English scale guide](#scale-and-memory-start-at-1x).

`scale` 同时乘输入的宽和高，像素数及完整 IMAGE 输出内存大致按 **倍率的平方** 增长，帧数翻倍还会再翻倍。以 **1472×2688、360 帧、RGB float32** 为例：

| `scale` | 实际输出宽×高（向上对齐 128） | 像素数约为输入的 | 仅完整 IMAGE 输出所需 RAM |
|---|---|---|---|
| **1×（默认）** | 1536×2688 | 1 倍 | **16.61 GiB** |
| 2× | 2944×5376 | 4 倍 | **63.68 GiB** |
| 4× | 5888×10752 | 16 倍 | **254.71 GiB** |

表中不含输入帧、模型、缓存和其他程序，因此 64GB 内存也无法容纳上述 2× 的完整处理过程。这是输出张量的计算值，不是显存或速度实测。

- **内存紧张先用 1×，高分辨率输入可先试 0.5×**，确认短视频成功后再试 1.5× / 2× / 4×。1× 仍运行模型，并向上对齐 128；不是跳过处理，也不保证任意片长都能运行。
- **修改倍率时，将 `height` 和 `width` 都设为 0。** 显式宽高会覆盖 `scale`，仅调低倍率不会改变这些固定输出尺寸。
- **已经放大过的视频要按当前输入尺寸计算。** 上游已做 2×，本节点再设 2×，相对最初输入就是约 4× 宽高、16 倍像素。
- **长视频优先用 Streaming VIDEO → SaveVideo。** 它避免完整 IMAGE 输出占满 RAM，但磁盘 tile 暂存仍随帧数、输出面积和重叠增加；tiling / offload 主要降低显存，不能消除普通 IMAGE 批次的 RAM。
- **已有画布保留原来的 2× / 4×。** 更新插件后，新建节点或重新导入新版示例才会使用 1×；已有工作流请在画布手动改 `scale` 并保存。程序修改 JSON 时同步 `widgets_values` 与 `widgets_values_named`，随后重新导入。

## 中文快速开始

1. 按下方命令克隆到 `ComfyUI/custom_nodes/ComfyUI-FlashVSR_Plus`，用 **ComfyUI 自己的 Python** 安装依赖，然后重启 ComfyUI。
2. 视频优先将 [streaming_video.json](example_workflows/streaming_video.json) 拖入画布并选择自己的视频；需要 IMAGE 帧输出时使用 [native_video.json](example_workflows/native_video.json)。旧版 ComfyUI 或 VHS 用户使用 [image_batch.json](example_workflows/image_batch.json)。
3. Loader 先用 `mode=turbo`、`attn_backend=auto`、`offload=auto`、`decoder=tiny`。默认 `profile=auto`，24GB 级显卡优先完整空间推理；显存压力较大时可用新增的 balanced 工作流。`auto_download=true` 会下载所需模型；手动安装只需下表中 **Turbo 的两个文件 + TCDecoder.ckpt**。
4. Upscale 先用短视频测试，保留低内存起步的默认值 `scale=1.0`、`height=0`、`width=0`，以及 `tiled=auto`、`temporal_compression=4`、`cache_offload=auto`、`decode_every=4`、`oom_retry=true`。确认资源充足后再增加片长与尺寸；倍率说明见上方重点提示。
5. Streaming 节点的 `video` 直接接 `SaveVideo`。普通 IMAGE 输出路径将 `images/audio/fps` 接到 `FlashVSR+ Create VIDEO`，再接 `SaveVideo`。IMAGE/VHS 路径的接法见工作流章节。

**模型目录是 `ComfyUI/models/FlashVSR_Plus/`，不是插件目录。** 如果你的 ComfyUI 位于 `C:\Comfyui`，模型就放在 `C:\Comfyui\models\FlashVSR_Plus\`。已有完整权重的短输入执行与速度测试，尚未验证整段长视频的画质、峰值显存和速度，详见[验证记录](docs/VALIDATION.md)。

## 中文：先了解默认行为

同一个插件提供完整空间、均衡与低显存策略，共用官方代码与权重。`profile=auto` 在加载和每次推理时检测当前 CUDA 设备的总显存与空闲显存。24GB 级设备按 **总容量 ≥23GiB** 识别，兼容 Windows 上常见的 23.9GiB 上报值。

| 情况 | 自动策略 | 空间处理 | 权重卸载 / 解码 |
|---|---|---|---|
| 24GB 级或以上、空闲 ≥18GiB，稀疏后端可用 | `official` | 不做 DiT 空间分块 | 解码时卸载 DiT，模型阶段卸载 decoder；KV auto |
| 不满足上一行，但总容量 ≥15GiB、空闲 ≥12GiB | `balanced` | 512 像素分块；空闲 ≥16GiB 时用 768 | 任务内权重驻留、任务结束卸载；KV / TinyDecoder auto |
| 其余较小或繁忙显卡 | `low_vram` | 256 / 384 / 512 像素重叠分块 | DiT block offload、KV always、串行 TinyDecoder |
| 只有 SDPA，且目标面积 >512×512 | `low_vram` | 最多 384 像素分块 | 避免 SDPA 大尺寸稀疏掩码的内存开销 |

`profile=official/balanced/low_vram` 可以手动指定策略；`tiled=enabled/disabled` 可以覆盖空间分块决定。`tiled=auto` 使用自动 tile 参数，忽略手动 `tile_size/tile_overlap`。`offload=auto` 使用策略默认值。手动 `balanced` 在空闲不足 12GiB 时也会使用 block offload 和保守分块；显存仍不足时保留 OOM 重试。

Turbo 严格保留官方 **Hybrid-B25**：第 0/6/12/18/24 层使用原始稀疏注意力，其余 25 层使用局部时空注意力。Base 与 V1 按官方配置关闭 Hybrid 替换。**RY-DMD 是训练/蒸馏方法，效果来自官方已训练权重，没有额外的推理开关。**

空间 tiling 会限制跨 tile 的全局交互，因此与官方不分块结果存在画质差异。24GB 以下跑 1080p/2K 是本插件的优化目标，实际是否能运行取决于 GPU、其他进程、tile 大小和模式；没有固定显存或速度保证。

### 生成速度慢：先检查卸载方式和分块数

**低内存起步保留 `scale=1.0`，但不必在显存充足时强制 `offload=block`。** 逐层卸载每个时间块都要搬运 30 层权重，256/512 小 tile 也需要多次处理整段视频。`tiled=auto` 时，直接修改 `tile_size` 不生效，须改为 `tiled=enabled` 才使用手动值。

24GB 级显卡先用 [streaming_video.json](example_workflows/streaming_video.json) 的 `turbo / auto / auto / auto / tiny`，优先整帧；显存压力较大时用 [streaming_video_balanced.json](example_workflows/streaming_video_balanced.json) 的 `turbo / balanced / auto / auto / tiny`，Upscale 保持 **1×、`tiled=auto`、`cache_offload=auto`、`decode_every=4`、`oom_retry=true`**。balanced 在空闲显存 ≥16GiB 时使用 768/128 分块及 `resident` 权重策略。例如 1536×2688 输出从 512/128 的 **28 个 tile** 减少到 768/128 的 **12 个 tile**；模型权重在整次请求中驻留 GPU，完成、报错或取消后卸载。原始注意力、时序缓存与解码计算保持官方实现。

| 完整 Turbo 权重短测试 | `offload=model` | `offload=resident` | 驻留后的吞吐提升 |
|---|---|---|---|
| 17 帧、512×512、一个 tile | 3.61 秒 / 3.83GiB | 2.70 秒 / 6.99GiB | 约 34% |
| 17 帧、768×768、一个 tile | 6.52 秒 / 8.44GiB | 4.68 秒 / 11.62GiB | 约 39% |

测试环境：RTX 5090 Laptop 24GB、Torch 2.10、官方 Triton、BF16、固定输入/seed、相同 decoder auto；均预热后单次测量，时间包含任务结束清理，显存为 PyTorch peak allocated。两组对照输出逐元素一致。**这不是整段视频、包含文件编解码的计时或跨显卡速度保证。** 变更 tile 大小本身会改变可见空间上下文，不保证与旧 tile 输出逐元素一致。

同一显卡还完成了 **17 帧、1536×2688 输出、完整 Turbo 权重、不分块**的合成输入测试：31.89 秒，PyTorch peak allocated **12.88GiB**，无 OOM 重试。这个单次测试未做相同尺寸的预热，也不包含模型加载与文件编解码，不能直接当作 360 帧视频耗时。显存允许时优先整帧可避免重复 tile 的上下文与推理工作；balanced 用于整帧不适合的任务。

24GB+ 且显存充足时，也可使用 `profile=auto` 尝试官方不分块路径；查看 `info.tile_count`、`offload`、`dit_resident_during_decode`、`elapsed_seconds` 和 `oom_retries`。`elapsed_seconds` 统计推理尝试，未计入 Loader、原生视频输入准备及最终权重清理；上表使用外部计时。首次执行还包括模型加载与 Triton 编译，后续队列请复用同一个 PIPE。升级后重启并重新导入工作流；旧 PIPE 已安装 block 适配器时，须改变 Loader 参数或 `reload_token` 重新加载才能恢复更快策略。

## 文件结构

```text
ComfyUI-FlashVSR_Plus/
├── __init__.py                 # ComfyUI 注册
├── nodes.py                    # Loader / Upscale / ToVideo / Unload / Diagnostics
├── pipeline.py                 # PIPE 生命周期、官方流式调用、OOM 重试
├── profiles.py                 # 显存策略、真实 CUDA 后端检查
├── memory.py                   # DiT block / decoder offload、任务内权重驻留
├── video_streaming.py          # 连续时序推理、磁盘 tile 暂存、逐帧编码
├── frames.py                   # IMAGE / VIDEO、官方 bicubic 预处理
├── tiling.py                   # 空间网格及重叠融合
├── models.py                   # 自动下载、模型路径、V1 文件名映射
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE                     # Apache-2.0
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── assets/posi_prompt.pth       # 官方固定上下文，随插件提供
├── docs/
│   ├── ARCHITECTURE.md
│   └── VALIDATION.md
├── example_workflows/
│   ├── image_batch.json        # 可拖入的 ComfyUI 工作流
│   ├── native_video.json       # LoadVideo → Upscale → VIDEO → SaveVideo
│   ├── streaming_video.json    # LoadVideo → Streaming Upscale → SaveVideo
│   ├── streaming_video_balanced.json # 分块 + 任务内权重驻留备选
│   ├── pipe_reuse.json         # 一个 PIPE 供两次推理使用
│   ├── image_batch_api.json    # /prompt API 格式
│   ├── native_video_api.json
│   ├── streaming_video_api.json
│   └── streaming_video_balanced_api.json
├── scripts/
│   ├── check_install.py
│   ├── benchmark_speed.py      # 本地完整权重短测试，不下载模型
│   ├── make_workflows.py
│   └── vendor_upstream.py
├── tests/
│   ├── conftest.py
│   ├── test_contracts.py
│   ├── test_streaming.py
│   ├── test_pipeline.py
│   └── test_cuda.py
└── vendor/
    ├── __init__.py
    ├── manifest.json           # 官方 commit、文件 SHA256
    └── runtime/
        ├── __init__.py
        ├── flashvsr_plus/      # 官方 Python API
        ├── inference/          # 官方 pipeline / runner / decoder / preprocessing
        ├── model_io/           # 官方 checkpoint 加载
        ├── models/             # 官方 DiT / Hybrid Attention / Wan VAE
        └── flashvsr_attn/      # 官方 Triton 稀疏算子
```

## 中文安装

需要 NVIDIA CUDA 显卡和 BF16 支持。官方参考平台为 Linux / Python 3.10 / PyTorch 2.5–2.7。此插件保留 ComfyUI 自己的 PyTorch，不自动安装/降级 Torch。Windows 与更新的 Torch 需要匹配自己的 Triton/二进制依赖，具体已测范围见 [VALIDATION.md](docs/VALIDATION.md)。

1. 将仓库克隆到 `ComfyUI/custom_nodes/ComfyUI-FlashVSR_Plus/`；也可在 GitHub 下载 ZIP，解压后确保该目录下直接能看到 `__init__.py`，不要多套一层文件夹。
2. 使用 **ComfyUI 实际使用的 Python** 安装依赖。
3. 重启 ComfyUI，搜索 `FlashVSR+`。先执行 `GPU / Backend Check`，再运行示例。

### Windows Desktop / venv

以下以 ComfyUI 用户目录为 `C:\Comfyui`、Python 环境为 `.venv` 举例。Desktop 用户目录和 Python 路径可不同，请替换为实际位置。在 PowerShell 执行：

```powershell
Set-Location C:\Comfyui
git clone https://github.com/kkget/ComfyUI-FlashVSR_Plus.git .\custom_nodes\ComfyUI-FlashVSR_Plus
& .\.venv\Scripts\python.exe -m pip install -r .\custom_nodes\ComfyUI-FlashVSR_Plus\requirements.txt
```

### Windows Portable

在 `ComfyUI_windows_portable` 根目录执行；ComfyUI 的便携 Python 目录名是 `python_embeded`：

```powershell
git clone https://github.com/kkget/ComfyUI-FlashVSR_Plus.git .\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Plus
& .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Plus\requirements.txt
```

### Linux

在 ComfyUI 根目录、激活其 Python 环境后执行：

```bash
git clone https://github.com/kkget/ComfyUI-FlashVSR_Plus.git custom_nodes/ComfyUI-FlashVSR_Plus
python -m pip install -r custom_nodes/ComfyUI-FlashVSR_Plus/requirements.txt
```

插件已附带官方源码快照，无需单独 `pip install flashvsr-plus`，也无需将上游源码加入全局 `PYTHONPATH`。

### 更新

在 ComfyUI 根目录执行，随后用同一个 Python 重新安装 requirements 并重启 ComfyUI。已安装时无需再次 clone。

```bash
git -C custom_nodes/ComfyUI-FlashVSR_Plus pull --ff-only
python -m pip install -r custom_nodes/ComfyUI-FlashVSR_Plus/requirements.txt
```

Windows 用户将上面的 `python` 换成对应 `.venv\Scripts\python.exe` 或 portable 的 `python_embeded\python.exe`，并相应调整路径。ZIP 安装可用新版本替换插件目录；模型位于独立的 `models/FlashVSR_Plus/`。

### 两条注意力安装路径

**易安装路径：** 只安装 requirements，`attn_backend=auto` 会使用已可用的 BSA / 内置 Triton / SDPA。没有 Triton 时使用 PyTorch SDPA，自动降低 tile 大小，不编译扩展。

**推荐的完整空间路径：** 安装与 ComfyUI Python/Torch/CUDA 匹配的 Triton。Linux 可使用 `python -m pip install triton`；Windows 可使用兼容的 `triton-windows` wheel：

```powershell
# 仅在环境尚未提供 Triton 时安装；版本需匹配现有 Torch。
python -m pip install triton-windows
```

本插件内置的官方 Triton 算子只导入 `torch`、`triton` 与算子自身模块，不需要额外安装 FlashAttention，也不需要 nvcc 编译自定义 CUDA 扩展。首次运行会 JIT 编译 Triton kernel。

也可自行按 [MIT Han Lab BSA](https://github.com/mit-han-lab/Block-Sparse-Attention) 说明安装 `block_sparse_attn`。自动优先级与官方一致：**BSA → Triton → SDPA**。插件会实际运行一个小 CUDA 调用，检查导入成功但二进制/JIT 不可用的情况。显式选定后端失败时直接报错；`auto` 则尝试下一个后端。

不提供未经验证的 Sparse_SageAttention 替换：它不能通过改一个函数名就保证满足本模型的 block-sparse mask 语义。当前提供官方 Triton 与 SDPA 两种备选。

## 模型下载与目录

Loader 默认 `auto_download=true`：先检查本地文件，只在所选模式的 DiT、LQ projection 或 decoder 缺失时调用 Hugging Face 下载。**本地文件齐全时不联网，也不发送 HEAD 检查。** 下载使用 Hugging Face 的缓存与续传机制，随后原子复制到模型目录。下载保留 Hub 缓存，会额外使用磁盘空间。自动下载使用固定 revision，见 `models.py`；支持标准 HF 缓存、身份认证与 `HF_ENDPOINT` 镜像配置。

默认模型根目录为 `ComfyUI/models/FlashVSR_Plus/`。可通过 Loader 的 `model_directory` 或 ComfyUI 的 `FlashVSR_Plus` extra model path 指定其他目录。不同模式共享 decoder 与 prompt。

`model_directory` 指向下面目录树的根目录，例如 `C:\Comfyui\models\FlashVSR_Plus`，不要指到 `turbo` 子目录。Portable 对应 `ComfyUI_windows_portable\ComfyUI\models\FlashVSR_Plus\`。

```text
ComfyUI/models/FlashVSR_Plus/
├── turbo/
│   ├── dit.safetensors
│   └── lq_proj_in.safetensors
├── base/
│   ├── dit.safetensors
│   └── lq_proj_in.safetensors
├── v1/
│   ├── dit.safetensors
│   └── LQ_proj_in.ckpt
├── TCDecoder.ckpt              # 默认 tiny 路径
├── Wan2.1_VAE.pth              # 仅 decoder=vae 时需要
└── prompt_tensor/
    └── posi_prompt.pth         # 从插件 assets 自动复制，无需下载
```

### 手动下载：精确文件与保存路径 / Manual downloads

以下全部为固定版本的文件直链，与自动下载一致。表中保存路径相对于 `ComfyUI/models/FlashVSR_Plus/`；GB/MB 使用十进制近似值。只下载使用的模式即可。

| 模式 / Mode | 下载文件 / Download | 保存路径 / Local path | 大小 / Size |
|---|---|---|---|
| **Turbo 必需** | [dit.safetensors](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/turbo/dit.safetensors?download=true) | `turbo/dit.safetensors` | 5.68 GB |
| **Turbo 必需** | [lq_proj_in.safetensors](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/turbo/lq_proj_in.safetensors?download=true) | `turbo/lq_proj_in.safetensors` | 1.15 GB |
| Base 必需 | [dit.safetensors](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/base/dit.safetensors?download=true) | `base/dit.safetensors` | 5.68 GB |
| Base 必需 | [lq_proj_in.safetensors](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/base/lq_proj_in.safetensors?download=true) | `base/lq_proj_in.safetensors` | 1.15 GB |
| V1 必需 | [diffusion_pytorch_model_streaming_dmd.safetensors](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/27561b186ded3402d7c975f4fd722e2885b6135f/diffusion_pytorch_model_streaming_dmd.safetensors?download=true) | `v1/dit.safetensors`（改名 / rename） | 5.68 GB |
| V1 必需 | [LQ_proj_in.ckpt](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/27561b186ded3402d7c975f4fd722e2885b6135f/LQ_proj_in.ckpt?download=true) | `v1/LQ_proj_in.ckpt` | 576 MB |
| **所有模式：decoder=tiny 必需** | [TCDecoder.ckpt](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/TCDecoder.ckpt?download=true) | `TCDecoder.ckpt` | 189 MB |
| 所有模式：仅 decoder=vae | [Wan2.1_VAE.pth](https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/Wan2.1_VAE.pth?download=true) | `Wan2.1_VAE.pth` | 508 MB |

默认 **Turbo + tiny = 第 1、2、7 行**，合计约 **7.02 GB**。无需下载 Base、V1 或 Wan VAE。`prompt_tensor/posi_prompt.pth` 已随插件附带，Loader 自动复制。手动下载完成后可设置 `auto_download=false` 离线运行。

V1 也接受 `v1/diffusion_pytorch_model_streaming_dmd.safetensors` 原名，以及**已经正确转换**的 `v1/lq_proj_in.safetensors`。不要把 `.ckpt` 直接改名成 `.safetensors`。

### 其他磁盘 / Extra model paths

也可在 ComfyUI 根目录的 `extra_model_paths.yaml` 添加以下配置，重启后让 Loader 的 `model_directory` 留空。目录中仍须保留 `turbo/base/v1` 等相对结构：

```yaml
flashvsr_external:
  base_path: D:/AI/models
  FlashVSR_Plus: FlashVSR_Plus
```

该例的 Turbo 文件路径是 `D:/AI/models/FlashVSR_Plus/turbo/dit.safetensors`。`model_directory` 留空时，插件按注册顺序搜索所有 `FlashVSR_Plus` 模型路径，并包含当前 ComfyUI 的 `models/FlashVSR_Plus/`。每个文件都先在所有本地目录查找，全部找不到才下载到第一个目录；不同目录可共用 decoder。ComfyUI Desktop 的空共享目录排在第一位时，不会遮住后面的本地文件。显式填写 `model_directory` 时只搜索该目录。

**手动安装 / 离线运行：** 在 Loader 设置 `model_directory=C:\Comfyui\models\FlashVSR_Plus`（替换为实际路径）、`auto_download=false`。Turbo 使用 `mode=turbo`、`decoder=tiny`，只需上面的三个权重。缺少文件会立即显示已检查路径，不会尝试联网。如果已有模型却仍看到 `HEAD .../turbo/dit.safetensors` 重试，检查本次实际运行的 Loader 参数和日志中的 `FlashVSR+ model search directories` / `using model`；更新前的旧版插件只查第一个注册目录，需更新并重启 ComfyUI，或直接显式填写正确目录。

新版 ComfyUI 保存的工作流可能同时包含 `widgets_values` 和 `widgets_values_named`，并按命名参数恢复控件。用脚本修改 JSON 时需要同步更新这两处，避免离线开关和模型路径恢复旧值。修改磁盘上的 JSON 后应重新导入工作流；已经打开的画布不会自动采用文件修改。直接在画布上修改 Loader 并保存也可同步这些参数。

## 节点与使用

| 节点 | 用途 |
|---|---|
| `FlashVSR+ Load PIPE (Auto VRAM)` | 加载模型一次，输出可重复使用的 `FLASHVSR_PLUS_PIPE` 和策略说明 |
| `FlashVSR+ Upscale (IMAGE / VIDEO)` | 接 IMAGE 批次或 VIDEO，输出 IMAGE、原 AUDIO、FPS、原 PIPE 和推理信息 |
| `FlashVSR+ Upscale VIDEO (Streaming)` | VIDEO → VIDEO 文件；持续时序推理、磁盘暂存空间 tiles、逐帧合成编码，支持音频透传；避免完整 IMAGE 占满 RAM |
| `FlashVSR+ Create VIDEO` | 使用输出帧、FPS、可选 AUDIO 创建原生 VIDEO，再接内置 SaveVideo |
| `FlashVSR+ Unload PIPE` | 释放 CPU/CUDA 模型；接超分后的 images 确保执行顺序 |
| `FlashVSR+ GPU / Backend Check` | 检测总/空闲显存，并实际检查注意力后端，不下载模型 |

普通 Upscale 的 `images` 与 `video` **只接一个**。此节点使用当前 ComfyUI 的 `get_components()` 接口提取帧、音频和精确来源 FPS；若同时外接 AUDIO，以外接音频为准。Streaming 节点使用文件路径，见下文。旧版本不注册 Create VIDEO / Streaming VIDEO，Upscale 仍支持传统 IMAGE + AUDIO；用 VideoHelperSuite 保存即可。

模型默认驻留 CPU RAM，在需要时搬入显卡。PIPE 可跨多个 Upscale 节点、多个队列任务复用；ComfyUI 正常缓存 Loader 时不会重新读取权重。每次推理都会新建时序状态，前一条视频的 KV/decoder 缓存不会污染下一条。显式 Unload 后，增加 Loader 的 `reload_token` 重新加载。自动升级到 block offload 后，该 PIPE 会继续保留 block offload；要恢复更快的驻留策略，重新加载 PIPE。

### 关键参数

- `mode=turbo/base/v1`：在 Loader 选择，切换模式会重新加载对应权重。
- `scale`：**默认 1×，从低内存开始**；当 height/width 都为 0 时使用，同时乘宽高，结果向上对齐到 128。2× / 4× 分别约为 4 / 16 倍像素；例：480×270 输入、1× 得到 512×384，4× 得到 1920×1152。
- `height/width`：同时设置，覆盖 scale；必须均为 128 的正整数倍。1080 不合法，可用 1024 或 1152。预处理沿用官方中心裁剪 + bicubic，比例变化会裁掉边缘。
- `temporal_compression=4/2/1`：默认 4；较大运动可试 2/1，耗时增加。保持输入 FPS 和帧数，不是插帧。
- `tiled`：`auto` 按显存策略；`enabled` 使用手动 tile_size/overlap；`disabled` 请求完整空间推理。
- `tile_size/tile_overlap`：单位为目标输出像素，均按 128 对齐，tile ≥256、overlap < tile。分块覆盖 DiT、LQ projection 与当前 decoder，每个 tile 保留完整视频时序。
- `offload=auto/model/block/resident/disabled`：model 在解码阶段间与任务结束后卸载；block 逐个 DiT block 搬运、显存最低；**resident 在任务内保留权重，任务结束卸载，速度更快但显存更高**；disabled 还保留任务结束后的权重。后两者只避免权重搬运，不改变官方 KV/LQ/decoder 时序状态处理。
- `cache_offload=auto/always/never`：控制官方原始稀疏层的 KV cache；与权重 offload 独立。CPU RAM/PCIe 流量会增加。
- `decode_every=4`：每 4 个 latent 帧解码一次，限制输入预处理和 latent 缓冲；0 使用官方完整时间轴模式。TinyDecoder 推荐 4，VAE 路径固定完整时间轴。
- `color_fix`：官方 wavelet 低频颜色修正，默认关闭；`color_correction_levels=5`。
- `seed`：控制官方 CUDA latent 噪声。同一软件/设备/后端/参数有助于复现；不同 attention kernel 的浮点结果可能有差异。
- `vae_tiled/vae_tile_size/vae_tile_overlap`：仅 `decoder=vae` 有效，单位为输出像素，内部转换成官方 VAE latent 单位。TinyDecoder 使用串行/增量策略或整条 pipeline 的空间 tiles，不能套用 VAE 参数。
- `oom_retry`：CUDA OOM 后释放缓存与权重、重启整个请求，开启 block/KV offload、串行解码并减小 tile；最多重试 2 次。不会丢帧继续，也不会自动降低请求的输出尺寸。显式关闭 tiled 时，开启 oom_retry 仍允许 OOM 后使用分块；严格禁止分块时同时关闭 oom_retry。

### 示例工作流

将 `example_workflows/*.json` 拖入 ComfyUI。`*_api.json` 为 API prompt，不用于拖入画布。

| 文件 | 用途 | 依赖 |
|---|---|---|
| [image_batch.json](example_workflows/image_batch.json) | 单图或 IMAGE 批次 → 超分 → SaveImage | ComfyUI 内置节点 + 本插件 |
| [native_video.json](example_workflows/native_video.json) | 视频 → 超分 → 保存视频，透传音频/FPS | 提供原生 VIDEO API 的 ComfyUI + 本插件 |
| [streaming_video.json](example_workflows/streaming_video.json) | 视频 → 流式超分 → SaveVideo，默认 1× + auto，24GB 级优先整帧 | 原生 VIDEO API；Loader decoder=tiny |
| [streaming_video_balanced.json](example_workflows/streaming_video_balanced.json) | 1× + balanced，较大分块与任务内权重驻留 | 原生 VIDEO API；Loader decoder=tiny |
| [pipe_reuse.json](example_workflows/pipe_reuse.json) | 一个 Loader 供两个 Upscale 复用 | ComfyUI 内置节点 + 本插件 |
| [image_batch_api.json](example_workflows/image_batch_api.json) | IMAGE 路径的 `/prompt` 请求体 | API 示例，修改输入文件名后使用 |
| [native_video_api.json](example_workflows/native_video_api.json) | VIDEO 路径的 `/prompt` 请求体 | 原生 VIDEO API；修改输入文件名后使用 |
| [streaming_video_api.json](example_workflows/streaming_video_api.json) | 流式 VIDEO 的 `/prompt` 请求体 | 原生 VIDEO API；修改输入文件名后使用 |
| [streaming_video_balanced_api.json](example_workflows/streaming_video_balanced_api.json) | balanced 流式 VIDEO 的 `/prompt` 请求体 | 原生 VIDEO API；修改输入文件名后使用 |

#### 长视频 / 系统内存不足：Streaming VIDEO

拖入 `streaming_video.json`，选择视频，保留 `scale=1.0`、`height=width=0`，设置 Loader `decoder=tiny`，将 Streaming 节点的 `video` 输出直接接 `SaveVideo`。该路径沿用官方 `TemporalVideoStream → StreamRunner(segment_writer=...)`，每个空间 tile 在整段视频中保留连续的 KV/decoder 状态。它不把视频切成彼此独立的推理段，也不创建完整输出 IMAGE 批次。`decode_every` 至少为 4，VAE decoder 暂不支持此文件输出路径。

输入 VIDEO 先通过 ComfyUI 的 `save_to()` 准备临时文件，文件型 VIDEO 的裁剪/时间截取由原生 API 处理。每个 tile 的输出片段以 **RGB8** 暂存磁盘，再逐帧按重叠权重融合，编码成 H.264/yuv420p；这会增加一次 8 位量化，与 float32 IMAGE 路径存在细微差异。`crf=18` 为默认输出质量，`keep_audio=true` 保留来源音轨；不兼容 MP4 的音轨转 AAC。音频较短时仍保留全部视频帧。

临时文件位于 ComfyUI temp 目录，tile 暂存和输入副本会在成功、失败或取消后清理；返回的临时 VIDEO 保留到 ComfyUI 清理 temp，务必接 `SaveVideo` 持久保存。开始时会检查临时磁盘空间，空间不足可减小目标尺寸，或在启动参数中用 `--temp-directory` 指向更大磁盘。内存不再按完整视频输出增长，但空间 tiles 的原始 RGB8 暂存大小仍与帧数和目标面积成正比，运行时间也会随 tile 数增加。

例：360 帧、1472×2688 的视频再放大 4×得到 5888×10752，float32 IMAGE 输出本身就需要 **254.71 GiB RAM**。仅改小 GPU tile 无法解决这个问题。Streaming 可以避免完整 IMAGE 分配，但 4×目标仍可能需要超过 100 GiB 临时磁盘。运行前先确认输入是否已被其他超分节点放大，避免重复放大。

#### 原生 VIDEO：加载并保存视频

1. 拖入 `native_video.json`，在内置 `LoadVideo` 选择输入视频；示例中的 `example.mp4` 只是占位文件名。
2. `LoadVideo.VIDEO` 接 Upscale 的 `video`，Loader 的 `pipe` 接 Upscale 的 `pipe`；Upscale 的 `images` 输入留空。
3. Upscale 的 `images`、`audio`、`fps` 分别接 `Create VIDEO` 的同名输入，再把它的 `VIDEO` 输出接 `SaveVideo`。原视频没有音频时也可运行。
4. 设置模式与目标尺寸，点击运行。输出由 `SaveVideo` 按其 filename prefix 保存到 ComfyUI 输出目录。

#### IMAGE 批次 / VideoHelperSuite

`image_batch.json` 默认使用内置 `LoadImage`，便于最小化测试。处理传统视频帧时，将其换成 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) 的视频加载节点：

- 加载节点的 `IMAGE` 帧批次 → Upscale.`images`，同时使 Upscale.`video` 留空。
- 加载节点的 `AUDIO` → Upscale.`audio`（可选）。把 Upscale 的 `fps` widget 转成输入，连接视频信息节点的实际加载 FPS，或手动填入该值。
- Upscale.`images` 与 `audio` → VHS Video Combine；该保存节点的 frame rate 也设置为相同 FPS。IMAGE 本身没有帧率元数据，插件不会从 IMAGE 猜测 FPS。
- 只需输出图片时，直接使用示例中的 `SaveImage`。

#### PIPE 复用 / 批量

`pipe_reuse.json` 演示一个 Loader 连接两个 Upscale；也可串接前一个 Upscale 的 `pipe` 输出，让后续节点继续复用同一份权重。普通队列重复运行时，保持 Loader 参数不变，ComfyUI 可复用缓存的 PIPE。GPU 执行会串行化，每段视频各自初始化时序缓存。

需要释放权重时，把最后一个 Upscale 的 `pipe` 和 `images` 接入 `Unload PIPE`。下次推理应增加 Loader 的 `reload_token`，避免复用已释放的 PIPE。

### 推荐设置

| 目标 | Loader | Upscale |
|---|---|---|
| 自动选择 | `turbo / auto / auto / auto / tiny`（mode/profile/backend/offload/decoder） | `tiled=auto`、`cache_offload=auto`、`decode_every=4` |
| 均衡速度与显存（balanced 示例） | `turbo / balanced / auto / auto / tiny` | 1×、`tiled=auto`；显存足够时 768/128 + resident，否则缩小 tile / block offload |
| 24GB+，优先完整空间推理 | `profile=auto`，安装可用的 Triton 或 BSA | 保持 `tiled=auto`，查看 `info` 确认是否实际分块 |
| 显存紧张，优先易安装 | `profile=low_vram`、`offload=block`、`decoder=tiny`；无 Triton 可选 SDPA | `tiled=enabled`、`tile_size=256` 或 `384`、`tile_overlap=128`、`cache_offload=always` |
| 必须禁止空间分块的对比测试 | `profile=official`，选可用的稀疏后端 | `tiled=disabled` 且 `oom_retry=false`；显存不足会报错 |

上述策略内置于同一个插件。自动选择会受**当前空闲显存**影响；24GB+ 显卡在被其他程序占用时也可能选择均衡或低显存策略。`info` 包含所选后端、策略、输出尺寸、tile、推理耗时与重试信息，可连接文本查看节点或查看日志。

**长视频的限制：** 普通 Upscale 的 IMAGE 输出仍需要完整 CPU 内存，原生 `VIDEO.get_components()` 也会解出完整帧集合。1920×1152 RGB float32 每帧约 25MiB，1000 帧仅最终输出就约 24.7GiB。插件会在输出最低内存需求超过当前可用 RAM 的 75% 时提前报出尺寸、帧数及估算，不会自动缩小结果。长视频保存为文件请使用上述 Streaming VIDEO；独立分段调用只复用权重，不延续时序状态。

## 常见问题

| 问题 | 处理方式 |
|---|---|
| 搜不到 FlashVSR+ / import failed | 确认 `custom_nodes/ComfyUI-FlashVSR_Plus/__init__.py` 存在；用启动 ComfyUI 的同一个 Python 安装 requirements，重启后查看启动日志中的异常。 |
| 没有原生 VIDEO 或 Create VIDEO 节点 | 当前 ComfyUI 没有相应 API 时只注册 IMAGE 路径；升级 ComfyUI，或改用 IMAGE/AUDIO + VHS。 |
| 找不到模型 | 检查 Loader 模式与上述完整路径；`model_directory` 指向模型根目录，不是 `turbo`。确认文件不是下载失败的 HTML 页面，也不是 ZIP/快捷方式。 |
| Hugging Face 下载失败 | 使用表中直链手动下载，然后关闭 `auto_download`；也可在启动 ComfyUI 前配置可用网络或标准 `HF_ENDPOINT`。插件不会擅自改用第三方镜像。 |
| 本地已有模型却仍在 HEAD / Retrying | 在 Loader 显式填写模型根目录并设 `auto_download=false`；确认本次工作流使用的参数。更新插件后重启，以启用多目录搜索；旧版只查第一个目录，Desktop 的共享目录可能排在本地目录前面。 |
| Triton 导入或 kernel 编译失败 | 使用匹配当前 Python/Torch/CUDA 的安装包，先运行 GPU / Backend Check；`auto` 会跳过失败后端，或显式选 `sdpa` + 小 tile。 |
| `skipping block_sparse: No module named block_sparse_attn` | 自动后端选择的信息日志；插件会继续尝试官方内置 Triton，再尝试 SDPA。查看后面的 `selected attention backend`。已有可用 Triton 时可显式选择 `triton_block_sparse`，无需安装 BSA。 |
| CUDA OOM / 24GB 仍分块 | 释放其他 GPU 进程占用；使用 low_vram、tiny、block offload、256/384 tile 与 always cache offload。自动重试已启用时最多重启请求两次。 |
| 生成很慢 / GPU 利用率低 | 检查是否强制 low_vram + block、tile 过小或后端为 SDPA。显存充足时使用 balanced + offload=auto + 官方 Triton，重载 PIPE；auto tiling 会忽略手动 tile_size。 |
| 输出高度不是 1080 | 模型要求宽高均为 128 的倍数；1920×1080 不合法，选 1920×1024 或 1920×1152。需要精确 1080p 时在超分后额外裁切/缩放。 |
| 色彩、构图或 tile 边缘有差别 | 预处理使用官方中心裁剪/bicubic，比例改变会裁边；可试 `color_fix`。空间 tiles 会改变全局上下文，可增加 tile 或在显存允许时关闭分块。 |
| 视频静音 / 速度不对 | IMAGE 输入需额外提供音频和真实 FPS，保存节点也使用该 FPS；原生 VIDEO 示例已连好透传。插件不插帧。 |
| PIPE has been unloaded | 增加 Loader 的 `reload_token` 后重新执行。 |
| 系统内存不足 / `DefaultCPUAllocator` | 使用 Streaming VIDEO 直接输出视频文件，或缩短帧数/降低输出尺寸。GPU offload 和 tiling 不会消除完整 IMAGE 输出所需 RAM。 |

诊断命令（激活 ComfyUI 环境后，从插件目录执行）：

```bash
python scripts/check_install.py --comfyui-path /path/to/ComfyUI
```

`--comfyui-path` 指包含 `comfy/` 和 `comfy_api/` 的 **ComfyUI 核心源码目录**；Desktop 安装中它可能与用户的 `models/custom_nodes` 目录不同。不提供这个参数也可检查 CUDA/后端，但原生 VIDEO 检测结果取决于核心是否已在 Python 搜索路径内。该命令不下载模型或启动 ComfyUI。

## English installation and usage

This is a community ComfyUI adapter using the official FlashVSR+ implementation, rather than an official OpenImagingLab extension. CUDA and BF16 support are required. The plugin preserves your existing Torch installation. Upstream's reference platform is Linux, Python 3.10, Torch 2.5–2.7; tested Windows/newer-Torch coverage is documented in [VALIDATION.md](docs/VALIDATION.md).

### Install and update

Clone from the ComfyUI root and install dependencies with **ComfyUI's own Python**, then restart ComfyUI:

```bash
# Activate the ComfyUI environment first; run from its root.
git clone https://github.com/kkget/ComfyUI-FlashVSR_Plus.git custom_nodes/ComfyUI-FlashVSR_Plus
python -m pip install -r custom_nodes/ComfyUI-FlashVSR_Plus/requirements.txt
```

For a Windows venv installation at `C:\Comfyui`, use `& C:\Comfyui\.venv\Scripts\python.exe` in PowerShell. For Windows Portable, run the following from the portable root:

```powershell
git clone https://github.com/kkget/ComfyUI-FlashVSR_Plus.git .\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Plus
& .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Plus\requirements.txt
```

To update, run `git -C custom_nodes/ComfyUI-FlashVSR_Plus pull --ff-only` from the ComfyUI root, reinstall requirements using the same interpreter, and restart. ZIP installations must contain `__init__.py` directly inside `custom_nodes/ComfyUI-FlashVSR_Plus/`. The upstream runtime is included; no separate `flashvsr-plus` installation is needed.

### Automatic VRAM profiles and attention

Run **GPU / Backend Check** first. `profile=auto` selects spatially untiled official inference on nominal 24GB+ cards (≥23GiB reported capacity) with ≥18GiB free, and overlapping spatial tiles otherwise. With only SDPA available, targets larger than 512×512 automatically use tiles. Manual profile, tiling and offload overrides are available. A nominal 24GB card reporting 23.9GiB is handled correctly.

Below the untiled threshold, cards with ≥15GiB total and ≥12GiB free automatically use `balanced`; smaller/busier cards use `low_vram`. You can explicitly select `balanced` on a 24GB card too. It uses 768/128 tiles with ≥16GiB free, 512/128 with ≥12GiB free, and conservative block offload below that. SDPA tiles stay capped at 384. Automatic policies are estimates backed by OOM recovery, not guaranteed memory bounds.

The easy installation uses existing PyTorch SDPA and requires no compiled attention extension. For faster full spatial inference, install a compatible `triton` package on Linux or `triton-windows` on Windows. The bundled upstream Triton operator does not import FlashAttention and does not require an nvcc extension build. Optional `block_sparse_attn` is also supported. Auto selection tries **BSA → bundled Triton → SDPA** and executes a tiny CUDA smoke test before selecting an optimized backend. Explicitly requested unavailable backends raise an error.

### Download and place models

The default Loader automatically downloads **only the selected preset and decoder** into `ComfyUI/models/FlashVSR_Plus/`. Files are downloaded through the Hugging Face cache and copied atomically, so additional cache disk space is needed. The download table above has pinned direct links, file sizes and exact local paths.

**The default Turbo + tiny setup needs only these three downloaded files (about 7.02 GB):**

```text
ComfyUI/models/FlashVSR_Plus/turbo/dit.safetensors
ComfyUI/models/FlashVSR_Plus/turbo/lq_proj_in.safetensors
ComfyUI/models/FlashVSR_Plus/TCDecoder.ckpt
```

For example, if ComfyUI is installed at `C:\Comfyui`, the first file belongs at `C:\Comfyui\models\FlashVSR_Plus\turbo\dit.safetensors`. Portable uses `ComfyUI_windows_portable\ComfyUI\models\FlashVSR_Plus\`. Set `auto_download=false` after manual installation. Leave `model_directory` empty for the default root, or point it at the directory containing `turbo/`, not at the `turbo/` subdirectory itself. The extra-model-path YAML example above can move all weights to another disk.

With an empty `model_directory`, the loader searches **all** registered `FlashVSR_Plus` directories plus the current ComfyUI `models/FlashVSR_Plus/` folder before downloading each missing file to the first directory. An empty Desktop shared folder cannot hide weights in another configured directory. An explicit `model_directory` restricts the search to that root. Complete local weights never trigger Hub requests, including HEAD, even with `auto_download=true`. For strict offline use, set `auto_download=false`; missing files immediately report every searched path. The log lists the search directories and the actual files selected.

Recent ComfyUI workflow JSON may contain both `widgets_values` and `widgets_values_named`, with named values used to restore controls. When editing JSON programmatically, update both representations so the offline switch and model root agree. Re-import the file after editing it; an already-open canvas does not reload changes from disk. Editing the Loader controls on the canvas and saving also keeps these values synchronized.

Turbo/Base and shared decoders come from [ruchangyao/FlashVSR_plus](https://huggingface.co/ruchangyao/FlashVSR_plus/tree/main). V1 comes from [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/tree/main): map `diffusion_pytorch_model_streaming_dmd.safetensors` to `v1/dit.safetensors`, and keep `v1/LQ_proj_in.ckpt` in its original format. The original V1 DiT filename is also accepted inside `v1/`. Never rename a pickle checkpoint to `.safetensors`. The fixed prompt is included and copied automatically. `Wan2.1_VAE.pth` is needed only for `decoder=vae`; the default official path uses `TCDecoder.ckpt`. Base/V1 are optional and not needed for Turbo.

### Quick workflow

1. Drag [native_video.json](example_workflows/native_video.json) into ComfyUI and replace `example.mp4` with your video. For an IMAGE batch or older ComfyUI, start with [image_batch.json](example_workflows/image_batch.json).
2. Set the Loader to `turbo / auto / auto / auto / tiny` (mode/profile/backend/offload/decoder) and enable automatic download, or install the three files above manually. Use the optional `streaming_video_balanced.json` for larger spatial tiles and within-request weight residency when full-frame inference does not fit.
3. Start with a short clip and the lower-memory defaults: `scale=1.0`, `height=width=0`, `tiled=auto`, temporal compression 4, decode interval 4 and OOM recovery enabled. Check available resources before increasing the scale or clip length.
4. The native VIDEO example connects Upscale's `images`, `audio` and `fps` to Create VIDEO, then SaveVideo. In the IMAGE example, replace LoadImage with your video frame loader as needed; connect audio separately and set both Upscale and your video saver to the actual loaded FPS. IMAGE tensors do not carry FPS metadata.
5. Reuse the Loader's PIPE for additional clips. Use [pipe_reuse.json](example_workflows/pipe_reuse.json) as an example. Keep Loader settings unchanged to allow ComfyUI to reuse its cached object.

Connect exactly one **IMAGE batch** or native **VIDEO** to Upscale. Outputs are CPU IMAGE frames, unchanged optional AUDIO, source FPS, the same reusable PIPE, and JSON information. Native VIDEO nodes are registered only when the installed ComfyUI provides the relevant API. On older ComfyUI use IMAGE/AUDIO and VideoHelperSuite.

Use a single Loader for multiple Upscale nodes or queue requests. Weights stay available in host RAM while offload releases CUDA allocations between requests. Every new clip starts fresh temporal state. After explicitly unloading a PIPE, change the Loader's `reload_token` to load again.

### Scale and memory: start at 1x

**All UI/API examples, both upscale nodes and the Python PIPE default to `scale=1.0`.** The Turbo/Base/V1 preset and automatic VRAM policy do not change your requested scale. Set **both `height=0` and `width=0`** for scale to take effect; explicit dimensions override it.

Scale multiplies width and height, so pixels and complete float32 IMAGE output RAM grow approximately with **scale squared**, and linearly with frame count. A 1472×2688 input with 360 frames needs **16.61 GiB at 1×** (1536×2688 after alignment), **63.68 GiB at 2×**, or **254.71 GiB at 4×** just for the output IMAGE tensor. Input frames, model weights and caches need additional memory. These are calculated host RAM figures, not measured VRAM or performance claims.

Try 1× first, or 0.5× for a large input, then increase to 1.5× / 2× / 4× after a short test. 1× still runs the model and rounds dimensions upward to multiples of 128; it does not guarantee enough memory for every clip. If another node has already enlarged your input 2×, another 2× here produces approximately 4× dimensions relative to the original. Prefer **Streaming VIDEO → SaveVideo** for long clips; its temporary disk usage still grows with frame count, target area and overlap. GPU tiling/offload cannot remove the RAM required for a complete IMAGE batch.

Updating the plugin does not reset widgets in existing workflows. Restart ComfyUI to load the new node definitions, then import the new examples or manually set `scale=1.0`, `height=width=0` and save. When editing JSON, synchronize positional `widgets_values` and `widgets_values_named`, then re-import the file.

### Important controls

| Parameter | Behavior |
|---|---|
| `mode` | Select Turbo, Base or V1 in the Loader; loads that preset's weights. |
| `scale`, `height`, `width` | Default scale 1×. Leave dimensions at 0 to use scale with upward 128 alignment; 2× / 4× mean about 4× / 16× pixels. Otherwise set both dimensions to positive multiples of 128, overriding scale. |
| `temporal_compression` | 4/2/1, default 4. Lower values may help motion and cost more computation. Output frame count/FPS are preserved. |
| `tiled`, `tile_size`, `tile_overlap` | `auto` uses policy values; `enabled` uses manual values in output pixels. Sizes align to 128, tile ≥256, overlap < tile. |
| `offload` | `model` releases weights between decode stages/requests; `block` also moves individual DiT blocks. `resident` keeps weights on CUDA within one request and releases them afterward. `disabled` also keeps weights between successful requests. `auto` follows the profile. |
| `cache_offload` | `auto/always/never` controls official sparse-layer KV cache offload, independently of model weights. |
| `decoder`, `decode_every` | Default tiny decoder, incremental decoding every 4 latent frames. 0 uses the full timeline. Wan VAE uses the full timeline. |
| `vae_tiled`, `vae_tile_size`, `vae_tile_overlap` | Native Wan VAE tiling, specified in output pixels; applies only to `decoder=vae`. |
| `color_fix`, `color_correction_levels` | Optional official wavelet color correction; default off, 5 levels. |
| `seed` | Seeds official latent noise. Results can differ across hardware/backends. |
| `oom_retry` | Up to two complete restarts with smaller tiles/stronger offload; preserves resolution and frame count. |
| `fps` | Used for IMAGE input; native VIDEO supplies its own FPS. |

Explicit height/width must both be divisible by 128. Leave both at zero to use scale; automatic dimensions round upward to 128. Preprocessing follows upstream RGB8 center crop and bicubic resize. Lower temporal compression to 2 or 1 for large motion. Temporal padding repeats the last frame internally and crops output to the exact original frame count.

`tiled=enabled` uses manual output-pixel tile/overlap values; `auto` uses the profile values. Spatial tiling reduces global context and may change quality. TinyDecoder uses bounded incremental/serial decoding; optional Wan VAE exposes native VAE tiling separately. RY-DMD is embodied in the pretrained weights, not an extra inference step. Turbo keeps upstream Hybrid-B25; Base/V1 keep their official non-Hybrid presets.

CUDA OOM recovery restarts the complete request with smaller tiles and stronger offload, up to two retries; it preserves the requested resolution and frame count. It can enable tiles even after `tiled=disabled`, unless `oom_retry=false` is also set. Actual resolution/VRAM limits depend on the workload and are not guaranteed.

Drag `image_batch.json`, `native_video.json`, or `pipe_reuse.json` into ComfyUI and select your own input files. Files ending in `_api.json` are `/prompt` API examples. These IMAGE-returning workflows pass audio/FPS through to Create VIDEO and SaveVideo and need CPU memory for the complete IMAGE output. Independent segments reuse weights but reset temporal history.

For long clips, use **`streaming_video.json`: LoadVideo → Upscale VIDEO (Streaming) → SaveVideo**. It requires `decoder=tiny`, numeric `decode_every>=4`, and defaults to **1×** with `height=width=0`. The official runner keeps continuous KV/decoder state across the entire timeline of each spatial tile. Its segment writer spools RGB8 tiles to disk, then composites/encodes one frame at a time to H.264/yuv420p. No complete IMAGE output is allocated; the 8-bit tile spool adds quantization compared with float32 IMAGE. `keep_audio=true` copies compatible audio or converts it to AAC; a short audio track never truncates video. File-backed native input is prepared through ComfyUI `save_to()`, including its trim/crop handling.

The node checks temporary disk space before inference and removes input/tile spools on success, failure and cancellation. The returned VIDEO is a temporary file: connect SaveVideo for persistent output. Set ComfyUI `--temp-directory` to a larger disk if needed. Disk usage grows with frame count, output area and tile overlap, even though output RAM does not grow with the full timeline. The standard IMAGE node now rejects a minimum output allocation exceeding 75% of available RAM, reporting the requested dimensions/frame count. Spatial GPU tiling cannot remove the RAM required by a complete IMAGE batch.

### Troubleshooting

**Speed settings:** retain `scale=1.0` and try `profile=auto` for untiled official inference on a free 24GB card, or the optional balanced example with `balanced / offload=auto / tiled=auto` before forcing block offload. With enough free VRAM it uses larger tiles, automatic KV residency and TinyDecoder auto, and holds model weights on CUDA across decode segments. Weights are released at the end of a request, on cancellation, or before OOM recovery. Only per-instance transfer methods are adapted within the request; official model/attention/temporal math is unchanged. Tiling itself still changes spatial context. `tiled=auto` ignores the manual tile size, and an existing block-offloaded PIPE must be reloaded to regain residency.

Full Turbo weights on an RTX 5090 Laptop, BF16 and official Triton: warmed 17-frame single-tile IMAGE tests took **3.61s → 2.70s at 512×512** and **6.52s → 4.68s at 768×768** when switching model-stage offload to resident weights, with identical output elements. Peak PyTorch allocations rose from 3.83 to 6.99GiB and from 8.44 to 11.62GiB respectively. These are single measurements including cleanup, excluding loading/JIT warmup and file codecs, not long-video throughput guarantees. Run `python scripts/benchmark_speed.py --model-directory /path/to/models/FlashVSR_Plus --size 512 --frames 17` to reproduce on your own GPU with existing local Turbo weights; the script does not download models or start ComfyUI. `info.elapsed_seconds` covers inference attempts only; Loader/input preparation and final weight cleanup are outside it.

- **Missing nodes:** check the plugin directory nesting, install requirements with the Python that starts ComfyUI, restart and inspect its import log. On older ComfyUI, use IMAGE/AUDIO + VideoHelperSuite instead of native VIDEO.
- **Missing weights/network failure:** follow the exact relative paths in the download table and disable `auto_download` for offline use. Standard Hugging Face networking/cache configuration and `HF_ENDPOINT` are supported; no third-party mirror is selected automatically.
- **HEAD retries despite local weights:** explicitly set the correct model root and `auto_download=false` in the workflow being executed. Older plugin versions searched only the first registered folder; update and restart ComfyUI to enable searching every local model directory.
- **Attention import/JIT failure:** use a Triton package compatible with the current environment, or select `sdpa` with small spatial tiles. Run GPU / Backend Check to inspect the actual selected backend.
- **`skipping block_sparse` message:** this is informational when using `auto`. The loader next tries bundled official Triton and then SDPA; check `selected attention backend`. BSA is optional.
- **CUDA OOM:** close other GPU workloads, select `low_vram`, `offload=block`, `decoder=tiny`, tile 256/384 with overlap 128 and `cache_offload=always`. This does not eliminate host RAM usage.
- **1080p dimensions:** 1920×1080 is not 128-aligned. Use 1920×1024 or 1920×1152, then crop/resize after super-resolution if an exact 1080p file is required.
- **Silent video/wrong speed:** connect audio and the actual FPS from your IMAGE video loader, and use the same FPS when saving. Upscale does not perform frame interpolation.
- **Unloaded PIPE:** increase Loader `reload_token`. **Long clips exhausting RAM:** use Streaming VIDEO or shorter/lower-resolution IMAGE batches.

Run `python scripts/check_install.py --comfyui-path /path/to/ComfyUI` from the plugin directory with the ComfyUI environment active. The path is the core source directory containing `comfy/` and `comfy_api/`; Desktop may store it separately from the user models folder. This command neither downloads weights nor starts the server.

## 开发检查 / Developer checks

```bash
python -m pip install pytest  # optional development dependency
python scripts/check_install.py --comfyui-path /path/to/ComfyUI
python -m pytest -q tests --rootdir=tests --confcutdir=tests
# Optional real CUDA attention and reduced-model execution checks:
python -m pytest -q tests --rootdir=tests --confcutdir=tests --run-cuda
```

`check_install.py` 不下载模型，也不启动或重启 ComfyUI。测试不评估完整预训练权重的画质；完整测试范围见 [VALIDATION.md](docs/VALIDATION.md)。

## Attribution

Official source snapshot: `a08fb184a84cc2157c4f271234555fcda7902c87`. Vendored Python files only change import namespaces; the manifest records hashes. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). UX references: [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR), [lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast). Their older FlashVSR pipelines are not used as the FlashVSR+ implementation.

Code: [Apache-2.0](LICENSE). Model weights retain their original model-card terms. Bug reports: [GitHub Issues](https://github.com/kkget/ComfyUI-FlashVSR_Plus/issues). Include your ComfyUI/Python/Torch/CUDA versions, GPU memory, node settings and relevant error log; remove personal paths, prompts and credentials before posting.
