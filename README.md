# Optimize-gemm-on-GPU-step-by-step

逐步优化单精度矩阵乘法（SGEMM）的项目：以 cuBLAS 作为正确性与性能基准，从共享内存分块的 baseline kernel 出发，依次引入向量化访存、软件流水、消除 bank conflict、消除冗余地址计算与分支、cp.async 异步拷贝等手段，性能逐步逼近 cuBLAS。

目前包含 11 个版本（`matmul_00` ~ `matmul_10`），每个版本通常只引入一类改动（个别版本是有代价的中间状态，详见下文「优化路线」），适合按编号对照学习。kernel 使用 CUDA Runtime API 编写，参考实现使用 cuBLAS。

## 环境要求

- CMake >= 3.24
- 支持 C++17 的编译器（MSVC / GCC / Clang）
- **NVIDIA CUDA Toolkit**（使用 CUDA Runtime API 与 cuBLAS）

## 安装依赖

1. 安装 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)。
   - Windows 默认安装路径：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<x.y>`
2. 确保 `nvcc` 在 `PATH` 中（Linux），或 `CUDA_PATH` 环境变量已设置（Windows 安装程序通常会自动设置）。

## 构建

```bash
cmake -B build -S .
cmake --build build --config Release
```

Windows 上若使用 Visual Studio 生成器，第二行等价于：

```bat
cmake --build build --config Release
```

如果 CUDA 不在默认位置，可传递：

```bash
cmake -B build -S . -D CUDAToolkit_ROOT=/usr/local/cuda
```

Windows 上 CUDA 通常通过 `CUDA_PATH` 环境变量被自动识别。

## 运行

程序接受 4 个参数：

```
matmul <M> <N> <L> <version>
```

- `M`、`N`、`L`：矩阵维度，计算 C(M×N) = A(M×L) × B(L×N)
- `version`：matmul 实现版本号，对应 `src/matmul.cu` 中 `matmul_funcs` 数组的下标，当前合法范围为 `[0, 10]`

程序交替执行 cuBLAS 参考实现与指定版本并输出 GFLOPS 对比，然后用 cuBLAS 的结果校验正确性，校验失败时退出码非零。

> **注意**：`matmul_09` 与 `matmul_10` 为非通用版本，要求 **M、N 能被 128 整除、L 能被 8 整除**（即 BM=BN=128、BL=8）。不满足时 kernel 不会启动，程序不会输出结果、正确性校验失败。4096 满足该条件。

### Windows

运行前需要把 CUDA 的动态库目录加到 `PATH`（除非该目录已在系统 PATH 中）：

```bat
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin

.\build\src\Release\matmul.exe 4096 4096 4096 0
```

想直接运行性能最好的版本（注意需满足上面的整除条件）：

```bat
.\build\src\Release\matmul.exe 4096 4096 4096 10
```

### Linux

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

./build/src/matmul 4096 4096 4096 0
```

## 优化路线

每个版本的实现位于 `src/matmul_<编号>.cu`，`version` 参数即对应编号。分块等参数（BM=BN=128、BL=8、RM=RN=8、VEC_SIZE=4）定义在 [include/config.h](include/config.h)。

- **matmul_00：baseline** — 共享内存分块
- **matmul_01：vectorize** — float4 向量化访存
- **matmul_02：pipeline** — 寄存器中转 + panel 双缓冲软件流水
- **matmul_03：optimize pipeline** — prologue/drain 外提，使其相互掩护，并消除内层条件分支
- **matmul_04：remove shared memory bank conflict** — area 分解；但引入了寄存器溢出和标量加载，是有代价的中间版本
- **matmul_05：remove local memory spill and restore vectorized epilogue load** — matmul_04 的 writeback 路径中，compiler 把 128-bit C 读拆成 stride-4 的标量循环；改进手段是写回循环完全展开，使访存恢复单条 LDG.128、C_reg 留在寄存器
- **matmul_06：remove redundant address computation** — 整数除法/基地址外提
- **matmul_07：remove redundant conditional branch** — 显式三段流水，热循环无边界判断
- **matmul_08：async ldgsts** — cp.async，全局→共享内存异步拷贝
- **matmul_09：non-general-purpose version** — 要求 M/N/L 整除 BM/BN/BL，删除全部边界处理与零填充；指令数下降但是寄存器数量增加，实际占用率下降，性能上的增益和损失基本互相抵消，是有代价的中间版本
- **matmul_10：#pragma unroll argument specified** — 使编译器编译出性能更优的指令；指令数高于 matmul_09 但低于 matmul_08，而寄存器数量下降，实际占用率恢复到 matmul_08 的水平，M=N=L=4096 下性能最接近 cuBLAS

## CMake 关键配置变量

| 变量 | 说明 |
|---|---|
| `CUDAToolkit_ROOT` | 指向 CUDA 安装根目录，用于 `find_package(CUDAToolkit REQUIRED)` |
| `CMAKE_CUDA_ARCHITECTURES` | 目标 GPU 架构，默认为 `native`（自动检测本机 GPU）。可用 `-D CMAKE_CUDA_ARCHITECTURES=86` 等值覆盖 |

## 常见问题

### 1. CMake 报错：`Could not find CUDAToolkit`

原因：CMake 没找到 CUDA Toolkit。

解决：
- 确认已安装 CUDA Toolkit。
- Windows：确认 `CUDA_PATH` 环境变量存在。
- Linux/macOS：确认 `nvcc` 在 `PATH` 中，或传递 `-D CUDAToolkit_ROOT=/usr/local/cuda`。

### 2. 运行时报错：找不到 `cudart64_xx.dll` / `cublas64_xx.dll`

原因：可执行文件找不到运行时动态库。

解决：按上面的“运行”章节把 CUDA 的 `bin`（Windows）或 `lib64`（Linux）目录加入环境变量。

### 3. 运行 `version` 9 或 10 时没有结果输出 / 校验失败

原因：这两个版本要求 M、N 能被 128 整除、L 能被 8 整除；不满足时 host 端直接返回，kernel 不会启动。

解决：换用满足整除条件的维度（如 4096），或改用 0~8 号通用版本。
