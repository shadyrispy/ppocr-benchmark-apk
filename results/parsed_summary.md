# PP-OCRv6 tiny — 多引擎最强配置基准 (forward + pipeline)

Devices: **S21U** = Galaxy S21 Ultra (SM-G998B, Exynos 2100, ARMv8.2, 有 fp16); **P20** = Huawei P20 Pro (EML-AL00, Kirin 970, ARMv8.0, 无 fp16).
Threads=4, iters=10, ABI=arm64-v8a. forward = 纯推理 (det/rec/cls 分离); pipeline = det 流水线 pre+forward+post.

### S21U 纯 forward 延迟 (mean_ms) + 数值校验

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

### P20 纯 forward 延迟 (mean_ms) + 数值校验

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

### S21U det 流水线延迟  (det pre+forward+post, threads=4, iters=10)

| engine/config | pre ms | fwd ms | post ms | pipeline ms |
|---|---|---|---|---|
| lw/cpu | 5.44 | 319.86 | 4.20 | 329.50 |
| ort/cpu | 5.57 | 49.70 | 4.18 | 59.45 |
| ort/int8 | 5.57 | 59.20 | 4.70 | 69.46 |
| ort/nnapi | ERR | ORT NNAPI EP segfaults on this device/OR | | |
| ort/nnapi-fp16 | ERR | ORT NNAPI EP segfaults on this device/OR | | |
| mnn/cpu | 5.46 | 54.13 | 4.10 | 63.69 |
| mnn/opencl | 5.46 | 55.40 | 4.06 | 64.91 |
| mnn/opencl-fp16 | 5.60 | 41.65 | 4.36 | 51.62 |
| mnn/vulkan | 5.52 | 54.81 | 4.26 | 64.60 |
| mnn/vulkan-fp16 | 5.48 | 42.24 | 4.33 | 52.06 |
| ncnn/cpu | 5.70 | 48.92 | 4.50 | 59.13 |
| ncnn/vulkan | 5.45 | 103.05 | 4.03 | 112.53 |
| ncnn/vulkan-fp16 | 5.47 | 72.80 | 11.51 | 89.78 |
| ncnn/int8 | 5.66 | 41.63 | 2.17 | 49.47 |

### P20 det 流水线延迟  (det pre+forward+post, threads=4, iters=10)

| engine/config | pre ms | fwd ms | post ms | pipeline ms |
|---|---|---|---|---|
| lw/cpu | 53.07 | 5611.89 | 15.94 | 5680.89 |
| ort/cpu | 52.93 | 522.06 | 15.00 | 589.99 |
| ort/int8 | 59.87 | 1288.88 | 17.02 | 1365.77 |
| ort/nnapi | ERR | ORT NNAPI EP segfaults on this device/OR | | |
| ort/nnapi-fp16 | ERR | ORT NNAPI EP segfaults on this device/OR | | |
| mnn/cpu | 53.14 | 416.31 | 12.09 | 481.55 |
| mnn/opencl | 53.13 | 416.20 | 12.09 | 481.41 |
| mnn/opencl-fp16 | 55.73 | 433.77 | 12.85 | 502.36 |
| mnn/vulkan | 59.74 | 458.62 | 13.60 | 531.96 |
| mnn/vulkan-fp16 | 60.21 | 455.53 | 13.66 | 529.40 |
| ncnn/cpu | 59.66 | 477.98 | 16.58 | 554.22 |
| ncnn/vulkan | 64.64 | 626.87 | 24.57 | 716.08 |
| ncnn/vulkan-fp16 | 63.67 | 344.39 | 32.24 | 440.30 |
| ncnn/int8 | 59.64 | 417.19 | 10.11 | 486.93 |

## 每引擎最强配置 (按 det 纯 forward mean_ms 排名，仅计入 numerics_ok)

### S21U
- **lw**: 最强 = `lw/cpu` (det 322.44 ms)
- **ort**: 最强 = `ort/cpu` (det 46.66 ms)
- **mnn**: 最强 = `mnn/opencl-fp16` (det 41.92 ms)
- **ncnn**: 最强 = `ncnn/int8` (det 44.76 ms)

### P20
- **lw**: 最强 = `lw/cpu` (det 6092.30 ms)
- **ort**: 最强 = `ort/cpu` (det 558.79 ms)
- **mnn**: 最强 = `mnn/vulkan` (det 443.74 ms)
- **ncnn**: 最强 = `ncnn/vulkan-fp16` (det 346.81 ms)
