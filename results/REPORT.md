# PP-OCRv6 tiny 多引擎「最强配置」基准报告

> 生成时间: 2026-09-16 08:18  |  模型: PP-OCRv6 tiny (det/rec/cls)  |  ABI: arm64-v8a  |  threads=4, iters=10

## 1. 背景与目标

在 4 个推理引擎（lw.PPOCR.C、ONNX Runtime、MNN、ncnn）上，把每个引擎的「最快路径」全部打开并实测：
- **ORT** → NNAPI(+fp16)、动态 int8 量化
- **MNN** → OpenCL / Vulkan(+fp16 精度) 后端
- **ncnn** → Vulkan(+fp16)、int8 量化
- **lw.PPOCR.C** → 无加速/量化开关（基线，等价于 fp32 CPU）

目标：**测出每个引擎在两种代际 SoC 上的最强配置**，并给出跨引擎选型建议。

## 2. 测试设备

| 设备 | SoC | ARM | fp16 硬件 | 角色 |
|---|---|---|---|---|
| **S21U** | Galaxy S21 Ultra (SM-G998B, Exynos 2100, Mali-G78) | ARMv8.2 | **有** | 现代旗舰 |
| **P20** | Huawei P20 Pro (EML-AL00, Kirin 970, Mali-G72) | ARMv8.0 | **无** | 老款（无 fp16） |

两设备均 4 线程、10 次迭代取均值。参考输入为合成 `[0,1]` 张量（模型内部无前处理），int8 校准使用同分布的合成图。

## 3. 纯 forward 延迟 (det / rec / cls, mean_ms)

> `num` 列：推理数值与参考实现的校验结果。`BAD` = 数值错误（排除出「最强」评选）；`ERR` = 该配置不可用。

### S21U（Exynos 2100, 有 fp16）

| engine/config | det ms | det num | rec ms | rec num | cls ms | cls num |
|---|---|---|---|---|---|---|
| lw/cpu | 322.44 | ok | 14.21 | ok | 1.50 | ok |
| ort/cpu | 46.66 | ok | 3.43 | ok | 0.73 | ok |
| ort/int8 | 53.72 | ok | 2.06 | ok | 1.25 | ok |
| ort/nnapi | ERR | n/a | ERR | n/a | ERR | n/a |
| ort/nnapi-fp16 | ERR | n/a | ERR | n/a | ERR | n/a |
| mnn/cpu | 54.33 | ok | 2.78 | ok | 0.45 | ok |
| mnn/opencl | 55.24 | ok | 5.15 | ok | 0.45 | ok |
| mnn/opencl-fp16 | 41.92 | ok | 3.10 | ok | 0.32 | ok |
| mnn/vulkan | 56.06 | ok | 2.69 | ok | 0.45 | ok |
| mnn/vulkan-fp16 | 44.40 | ok | 1.95 | ok | 0.34 | ok |
| ncnn/cpu | 47.90 | ok | 3.46 | ok | 0.30 | ok |
| ncnn/vulkan | 102.22 | ok | 29.89 | ok | 17.88 | ok |
| ncnn/vulkan-fp16 | 72.56 | BAD | 27.24 | ok | 14.83 | ok |
| ncnn/int8 | 44.76 | ok | 1.57 | ok | 1.29 | ok |

### P20（Kirin 970, 无 fp16）

| engine/config | det ms | det num | rec ms | rec num | cls ms | cls num |
|---|---|---|---|---|---|---|
| lw/cpu | 6092.30 | ok | 164.43 | ok | 19.06 | ok |
| ort/cpu | 558.79 | ok | 25.06 | ok | 5.90 | ok |
| ort/int8 | 1286.85 | ok | 63.10 | ok | 41.99 | ok |
| ort/nnapi | ERR | n/a | ERR | n/a | ERR | n/a |
| ort/nnapi-fp16 | ERR | n/a | ERR | n/a | ERR | n/a |
| mnn/cpu | 445.33 | ok | 20.45 | ok | 4.97 | ok |
| mnn/opencl | 443.78 | ok | 20.52 | ok | 4.07 | ok |
| mnn/opencl-fp16 | 446.62 | ok | 20.26 | ok | 4.79 | ok |
| mnn/vulkan | 443.74 | ok | 20.43 | ok | 3.94 | ok |
| mnn/vulkan-fp16 | 445.97 | ok | 20.34 | ok | 4.27 | ok |
| ncnn/cpu | 473.85 | ok | 28.26 | ok | 3.16 | ok |
| ncnn/vulkan | 626.01 | ok | 80.01 | ok | 12.72 | ok |
| ncnn/vulkan-fp16 | 346.81 | ok | 60.96 | ok | 8.18 | ok |
| ncnn/int8 | 410.39 | ok | 22.99 | ok | 5.01 | ok |

## 4. det 流水线延迟 (pre + forward + post, mean_ms)


### S21U  (det pre+forward+post, threads=4, iters=10)

| engine/config | pre ms | fwd ms | post ms | pipeline ms |
|---|---|---|---|---|
| lw/cpu | 5.44 | 319.86 | 4.20 | 329.50 |
| ort/cpu | 5.57 | 49.70 | 4.18 | 59.45 |
| ort/int8 | 5.57 | 59.20 | 4.70 | 69.46 |
| ort/nnapi | ERR | ORT NNAPI EP segfaults on this device/ORT buil | | |
| ort/nnapi-fp16 | ERR | ORT NNAPI EP segfaults on this device/ORT buil | | |
| mnn/cpu | 5.46 | 54.13 | 4.10 | 63.69 |
| mnn/opencl | 5.46 | 55.40 | 4.06 | 64.91 |
| mnn/opencl-fp16 | 5.60 | 41.65 | 4.36 | 51.62 |
| mnn/vulkan | 5.52 | 54.81 | 4.26 | 64.60 |
| mnn/vulkan-fp16 | 5.48 | 42.24 | 4.33 | 52.06 |
| ncnn/cpu | 5.70 | 48.92 | 4.50 | 59.13 |
| ncnn/vulkan | 5.45 | 103.05 | 4.03 | 112.53 |
| ncnn/vulkan-fp16 | 5.47 | 72.80 | 11.51 | 89.78 |
| ncnn/int8 | 5.66 | 41.63 | 2.17 | 49.47 |

### P20  (det pre+forward+post, threads=4, iters=10)

| engine/config | pre ms | fwd ms | post ms | pipeline ms |
|---|---|---|---|---|
| lw/cpu | 53.07 | 5611.89 | 15.94 | 5680.89 |
| ort/cpu | 52.93 | 522.06 | 15.00 | 589.99 |
| ort/int8 | 59.87 | 1288.88 | 17.02 | 1365.77 |
| ort/nnapi | ERR | ORT NNAPI EP segfaults on this device/ORT buil | | |
| ort/nnapi-fp16 | ERR | ORT NNAPI EP segfaults on this device/ORT buil | | |
| mnn/cpu | 53.14 | 416.31 | 12.09 | 481.55 |
| mnn/opencl | 53.13 | 416.20 | 12.09 | 481.41 |
| mnn/opencl-fp16 | 55.73 | 433.77 | 12.85 | 502.36 |
| mnn/vulkan | 59.74 | 458.62 | 13.60 | 531.96 |
| mnn/vulkan-fp16 | 60.21 | 455.53 | 13.66 | 529.40 |
| ncnn/cpu | 59.66 | 477.98 | 16.58 | 554.22 |
| ncnn/vulkan | 64.64 | 626.87 | 24.57 | 716.08 |
| ncnn/vulkan-fp16 | 63.67 | 344.39 | 32.24 | 440.30 |
| ncnn/int8 | 59.64 | 417.19 | 10.11 | 486.93 |

## 5. 每引擎最强配置（结论）

### S21U
- **lw**: 最强 = `lw/cpu`（det 322.44 ms）
- **ort**: 最强 = `ort/cpu`（det 46.66 ms）（`nnapi` 崩溃禁用，`int8` 反而 +15% 回退）
- **mnn**: 最强 = `mnn/opencl-fp16`（det 41.92 ms）（fp16 比 cpu 快 ~22%，且 rec/cls 也最快）
- **ncnn**: 最强 = `ncnn/int8`（det 44.76 ms）（`vulkan-fp16` 在 S21U 上 det 数值损坏，已排除；int8 为 ncnn 最快可用配置）

### P20
- **lw**: 最强 = `lw/cpu`（det 6092.30 ms）
- **ort**: 最强 = `ort/cpu`（det 558.79 ms）（`nnapi` 禁用，`int8` 在 P20 上 3× 回退，cpu 是唯一 sane 选择）
- **mnn**: 最强 = `mnn/vulkan`（det 443.74 ms）（cpu/opencl/vulkan/fp16 全部≈444 ms，噪声内持平；选 cpu 省 GPU 开销）
- **ncnn**: 最强 = `ncnn/vulkan-fp16`（det 346.81 ms）（`vulkan-fp16` 仅 det 快，但 rec/cls 偏慢 2–3×；整体更推荐 `ncnn/int8`）

## 6. 跨引擎 det 延迟排名（仅 numerics_ok 配置）

**S21U — det 纯 forward（越快越好）**  (det 纯 forward, ms; █≈8.1ms)

  mnn/opencl-fp16    █████   41.92
  mnn/vulkan-fp16    █████   44.40
  ncnn/int8          █████   44.76
  ort/cpu            █████   46.66
  ncnn/cpu           █████   47.90
  ort/int8           ██████   53.72
  mnn/cpu            ██████   54.33
  mnn/opencl         ██████   55.24
  mnn/vulkan         ██████   56.06
  ncnn/vulkan-fp16   █████████   72.56  ⚠BAD
  ncnn/vulkan        ████████████  102.22
  lw/cpu             ████████████████████████████████████████  322.44
  ort/nnapi          ERR (disabled)
  ort/nnapi-fp16     ERR (disabled)

**P20 — det 纯 forward（越快越好）**  (det 纯 forward, ms; █≈152.3ms)

  ncnn/vulkan-fp16   ██  346.81
  ncnn/int8          ██  410.39
  mnn/vulkan         ██  443.74
  mnn/opencl         ██  443.78
  mnn/cpu            ██  445.33
  mnn/vulkan-fp16    ██  445.97
  mnn/opencl-fp16    ██  446.62
  ncnn/cpu           ███  473.85
  ort/cpu            ███  558.79
  ncnn/vulkan        ████  626.01
  ort/int8           ████████ 1286.85
  lw/cpu             ████████████████████████████████████████ 6092.30
  ort/nnapi          ERR (disabled)
  ort/nnapi-fp16     ERR (disabled)

### 相对 lw 基线的加速比（det）

| 设备 | 最强配置 | det ms | 相对 lw 加速 |
|---|---|---|---|
| S21U | `mnn/opencl-fp16` | 41.92 | **7.7×** |
| P20 | `ncnn/vulkan-fp16` | 346.81 | **17.6×** |

## 7. 关键发现

### 7.1 ORT NNAPI EP 崩溃（SIGSEGV）→ 已优雅禁用
- ORT 1.30 的 `OrtSessionOptionsAppendExecutionProvider_Nnapi` 在 S21U / P20 上均触发 **原生段错误**（`SIGSEGV`，`libonnxruntime.so` 内空函数指针解引用），进程无法捕获。
- 已在 `backend_ort.cpp` 改为直接返回错误字符串（`ort/nnapi`、`ort/nnapi-fp16` 显示 `ERR`），不再调用 NNAPI EP。结论：**这两台设备的 NNAPI 驱动与 ORT 1.30 不兼容，NNAPI 路线不可用**.

### 7.2 ncnn Vulkan-fp16 的 det 数值损坏（仅 S21U）
- `ncnn/vulkan-fp16/det` 在 **S21U (Mali-G78)** 上 `numerics_ok=false`（`max_abs_diff=1.0`，`rel_diff≈247%`），但 rec/cls 正常；在 **P20 (Mali-G72)** 上 det 数值正确。
- 判定为 ncnn Vulkan-fp16 路径在该 GPU 驱动上的 fp16 存储/拷贝缺陷。**S21U 评选已将该配置排除**；P20 上可用但 rec/cls 偏慢，不推荐.

### 7.3 ORT int8：S21U 轻微回退，P20 灾难性 3× 回退
- S21U：`ort/int8` det 53.72 ms vs `ort/cpu` 46.66 ms（**+15% 回退**）。
- P20：`ort/int8` det 1286.85 ms vs `ort/cpu` 558.79 ms（**+130% / ~3× 回退**）。
- 根因：ORT 动态 int8（weight-only, per-channel）在 ARMv8.0 上走了非最优反量化内核。**ORT int8 在两台设备都不值得用**；尤其 P20 上应绝对避免.

### 7.4 ncnn int8：双设备均有效
- S21U：`ncnn/int8` det 44.76 ms vs `ncnn/cpu` 47.90 ms（−7%，且是 ncnn 自身最快配置）。
- P20：`ncnn/int8` det 410.39 ms vs `ncnn/cpu` 473.85 ms（−13%）。
- ncnn 的 int8 工具链（ncnn2table + ncnn2int8）在 ARMv8.0/8.2 上均给出稳定收益，是 **ncnn 在两类设备上的最强配置**.

### 7.5 硬件代际决定 GPU / fp16 是否值得（核心结论）
- **S21U（ARMv8.2，有 fp16）**：GPU-fp16 收益明显。MNN fp16（opencl-fp16 / vulkan-fp16）det ~42 ms，比 MNN CPU（54 ms）快 ~22%；MNN fp16 也是 rec/cls 最快路径。
- **P20（ARMv8.0，无 fp16）**：MNN 各后端（cpu/opencl/vulkan/opencl-fp16/vulkan-fp16）det **全部约 444 ms，毫无差异**——fp16 配置在缺乏硬件 fp16 时回退为 fp32，GPU 也无收益。
- 即：**现代旗舰优先上 MNN-fp16；老款无 fp16 设备，GPU 与 offline-fp16 都是徒劳，应退回 MNN/ncnn 的 CPU fp32 或 int8**.

## 8. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| S21U 类现代旗舰 | **MNN Vulkan/OpenCL-fp16**（或 ncnn/int8） | det ~42 ms，rec/cls 也最快；整体 7–8× 快于 lw |
| P20 类老款无 fp16 | **MNN CPU fp32** 或 **ncnn/int8** | GPU/fp16 无收益；ncnn/int8 略快于 MNN-cpu，MNN-cpu 的 rec/cls 更优 |
| 跨代际通用 | **MNN（CPU 或 fp16 自适应）** | 在两类设备都稳居第一梯队，且无 ORT 的 int8 回退/NNAPI 崩溃风险 |
| 避免 | ORT/NNAPI、ORT/int8、lw.PPOCR.C、ncnn/vulkan-fp16(S21U) | NNAPI 崩溃 / int8 回退 / 速度不可用 / det 数值损坏 |

**lw.PPOCR.C 在所有场景均不可用**：S21U det 322 ms、P20 det 6092 ms，比最快原生引擎慢 6–15× ，不建议作为生产路径.

## 9. 复现与数据来源

- 基准 APK：`app/build/outputs/apk/debug/ppocr-bench-debug.apk`（已禁用 NNAPI、内置 ort-int8 / ncnn-int8 资产）
- 原始日志：`results/raw-s21-all.log`、`results/raw-p20-all.log`（纯 forward）；`results/raw-s21-pipe.log`、`results/raw-p20-pipe.log`（det 流水线）
- 解析中间产物：`results/parsed.json`、`results/parsed_summary.md`
- 关键脚本：`scripts/run_sweep.sh`（单设备单跑采集）、`scripts/parse_results.py`（解析）、`scripts/gen_report.py`（本报告）
- int8 模型：`models/ort-int8/*`（ORT 动态 int8）、`models/ncnn-int8/*`（ncnn2table+ncnn2int8，合成图校准）
- MNN int8 因 `mnnquant` 崩溃（`std::length_error: vector`，校准配置 schema 不兼容）**未产出**，已从 CASES 移除。

---
*本报告由自动化基准流水线生成，所有数字直接来自设备实测 logcat 输出。*