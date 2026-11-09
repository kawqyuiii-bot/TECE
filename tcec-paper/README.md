# TCEC 论文复现

目标：用和论文相同的模型与基座复现 TCEC。量化器全部来自官方代码，这里只增加 TCEC 校正。


## 设计思路

论文说 TCEC 与基座正交，所以把系统分成两层：

```
官方基座（不改动）                TCEC 层（本目录）
Q-Diffusion / SVDQuant  ──μ̃──▶  1. 校准：同一 x_t 上比较量化与全精度输出，按 Eq. 17 算 K
官方校准、官方量化模型            2. 采样：每步按 Eq. 18 修正 x_{t-1}
官方 FID / PSNR 评估     ◀──x──  3. none 和 paper 共用噪声，可以配对比较
```

| 要点 | 做法 | 原因 |
| --- | --- | --- |
| K | Eq. 17 闭式解，每步每通道一个，λ1 用论文经验公式，γ = 1e-8 | 与论文一致 |
| 校准轨迹 | 沿量化模型的轨迹走，同一 x_t 再过一次全精度模型 | Eq. 17 比较的是同一输入下的两个输出 |
| 校正 | 第一步没有 t+1 项，之后减去当前项和上一步的项 | Eq. 18（m=1） |
| 有 CFG 时 | 用引导后的噪声算 K 和校正 | 进入 DDIM 更新的是引导后的噪声 |
| 接入方式 | CIFAR 用自己的采样循环；SDXL 继承 `DDIMScheduler` | 不修改官方代码，升级和核对都方便 |

## 与论文的对应关系

| | CIFAR-10（Table 9） | SDXL（Table 1） |
| --- | --- | --- |
| 脚本 | `run_cifar.sh` | `run_sdxl.sh` |
| 模型 | DDPM CIFAR-10（ema_cifar10） | stable-diffusion-xl-base-1.0 |
| 基座 | Q-Diffusion（BRECQ + AdaRound，W4A8 / W4A6） | SVDQuant（W4A4 分组 64、rank 32；W8A8 per-token 动态激活、逐通道权重、rank 16） |
| 采样 | DDIM 100 / 250，quad 时间步 | DDIM 50，512×512 |
| K 的校准 | 1024 张（论文未写，取主实验的规模） | 1024 条 COCO prompt |
| 评估 | 5 万张，IS、FID、sFID | 5K prompt，MJHQ / sDCI，FID、PSNR 等 |
| 论文目标（基线 → TCEC） | W4A8 DDIM-100 FID 4.92 → 4.79；W4A6 39.07 → 10.76 | W4A4 PSNR 21.0 → 21.9；W8A8 26.4 → 27.3 |

**论文没写明、由我假定的：** SDXL 的 CFG 取 5.0，CIFAR 的 K 校准规模取 1024，W4A6 校准时 `--sm_abit` 用默认 8。

**per-token：** 线性层输入在量化前展开为（token, 通道），分组 `[1, -1]` 下每个 token 有独立量程；卷积层没有 token，按样本计算。这是根据 DeepCompressor 源码读出来的，没有运行验证。

**sFID：** 用 OpenAI 的 ADM 评估脚本，参考集是 CIFAR-10 的 5 万张训练图。论文没有说明 Table 9 用的 sFID 实现，这里取最常用的版本，数值如与论文有系统偏差，先查这一点。

**已知偏差：** PixArt、SDXL-Turbo、Flux 和视频模型没有做。

## 环境

建议 Python 3.10、CUDA GPU、torch 2.5。不要用 2.6 以上：它默认 `weights_only=True`，旧检查点可能读不出来。

`constraints.txt` 固定了 torch 2.5.1 以及必须与它匹配的库（diffusers、transformers 等）。不要单独升级这些库：新版 diffusers 会调用 torch 2.6 才有的 `torch.accelerator`，导入时就报错。

```bash
./setup.sh
python tests/test_local.py
```

做 SDXL 时用 `./setup.sh sdxl`。SDXL 部分建议单独建环境，并准备足够放下两个 UNet 的显存，校准 K 时两个模型同时在 GPU 上。

`setup.sh` 固定了读过的提交（q-diffusion `715783d`，deepcompressor `69f3473`）。

## 第一步：小规模试跑

先确认流程通畅。这一步的 FID 没有参考意义。

```bash
N=1024 BS=64 ./run_cifar.sh 8 100
```

如果已从 Q-Diffusion 的 Google Drive 下载 W4A8 校准好的模型，可以跳过官方校准：

```bash
N=1024 BS=64 ./run_cifar.sh 8 100 /path/to/official_w4a8.pth
```

## 第二步：完整复现 Table 9

```bash
./run_cifar.sh 8 100
```

```bash
./run_cifar.sh 8 250
```

```bash
./run_cifar.sh 6 100
```

```bash
./run_cifar.sh 6 250
```

`torch-fidelity` 的 IS、FID 在 `runs/cifar/w4a{8,6}/s{100,250}/{none,paper}.metrics.txt`，ADM 的 IS、FID、sFID 在同目录的 `.adm.txt`。图片已生成的批次会自动跳过，中断后重新运行即可续跑。

**先看基线：** W4A8 DDIM-100 的 `none` 应接近 FID 4.92。差很多时先查基座，这时 TCEC 的结果没有意义。

## 第三步：SDXL

```bash
./run_sdxl.sh 4
```

```bash
./run_sdxl.sh 8
```

脚本依次执行：生成全精度参考图、收集校准数据、SVDQuant 量化并评估、校准 K、带 TCEC 评估。结果在 `third_party/deepcompressor/examples/diffusion/runs` 下各任务的 `results.json`。需要 Hugging Face 上的 SDXL 模型和评估数据集。

## 文件

| 文件 | 作用 |
| --- | --- |
| `tcec_core.py` | Eq. 17 和 Eq. 18，与基座无关 |
| `cifar_tcec.py` | CIFAR：校准数据、K、采样 |
| `sdxl_scheduler.py` | 带 TCEC 的 DDIM 调度器 |
| `sdxl_tcec.py` | 把调度器接入 DeepCompressor，校准 K |
| `configs/` | SDXL 模型配置和 W8A8 配置 |
| `eval_adm.py` | 打包图片，调用 ADM 脚本计算 sFID |
| `tests/` | 不依赖 GPU 和第三方代码的测试 |
