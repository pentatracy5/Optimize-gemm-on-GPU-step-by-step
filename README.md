# Optimize-gemm-on-GPU-step-by-step

逐步优化矩阵乘法（GEMM）的项目。使用 CUDA Runtime API 与 cuBLAS 作为参考实现，后续会逐步加入自定义 CUDA kernel。

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
- `version`：matmul 实现版本号，对应 `src/matmul.cpp` 中 `matmulFuncs` 数组的下标

程序输出 cuBLAS 参考实现与指定版本的 GFLOPS 对比，并用 cuBLAS 的结果校验正确性，校验失败时退出码非零。注意当前唯一的版本 `0`（`matmul_00`）是空实现占位，校验失败属预期行为。

### Windows

运行前需要把 CUDA 的动态库目录加到 `PATH`（除非该目录已在系统 PATH 中）：

```bat
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin

.\build\src\Release\matmul.exe 4096 4096 4096 0
```

### Linux

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

./build/src/matmul 4096 4096 4096 0
```

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
