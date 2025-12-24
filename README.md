# CUDA FMCW Radar Signal Processing Acceleration

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)
![Language](https://img.shields.io/badge/Language-C%2B%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-orange)

## 📖 项目简介 (Introduction)

本项目基于 **NVIDIA CUDA** 实现了 FMCW（调频连续波）雷达信号处理链路的并行加速。

通过利用 GPU 的高并发特性，本项目目标解决传统 CPU 在处理高分辨率、多通道雷达数据时的性能瓶颈，实现从原始 ADC 数据到点云生成的**实时处理**。

目前主要针对以下算法模块进行了 CUDA 优化：
- **Range FFT (1D-FFT)**：距离维傅里叶变换
- **Doppler FFT (2D-FFT)**：多普勒维傅里叶变换

## 🚀 功能特性 (Features)

- [x] **高性能 FFT**：使用 `cuFFT` 库进行批量 1D 和 2D FFT 运算。
- [x] **自定义 CUDA Kernel**：针对矩阵转置、数据加窗编写了高效的 Kernel。
- [x] **RAII**：基于现代C++，安全高效。
- [x] **模块化设计**：C++ 类封装，易于集成到现有的雷达 SDK 或 ROS 环境中。

## 🛠️ 信号处理流程 (Processing Pipeline)

该项目实现了标准的 FMCW 雷达信号处理流程：

1. **ADC Data Loading**: 加载原始雷达回波数据。
2. **Range Processing**:
   - Windowing (加窗)
   - FFT along fast-time axis (距离维 FFT)
3. **Doppler Processing**:
   - Transpose (矩阵转置)
   - FFT along slow-time axis (多普勒维 FFT)

## 📊 性能基准 (Performance Benchmark)

*测试环境: Intel Core 9 270H + NVIDIA RTX 5060 / Dataset: 100 Frames x 512 Chirps x 512 Samples x 4 RX*

| Module | CPU Time (ms) | GPU Time (ms) | Speedup |
| :--- | :---: | :---: | :---: |
| Range FFT + Doppler FFT | 832.222 | 137.314 | **6.06074x** |
| **Total Pipeline** | **832.222** | **137.314** | **6.06074x** |

## 🗓️ 开发优化计划 (Roadmap/To-Do)

后续计划完善完整的雷达点云生成链路，并针对GPU和现代C++特性进行优化：

- [ ] **CFAR Detection**：恒虚警率检测 (CA-CFAR / OS-CFAR)
- [ ] **Angle Estimation**：角度估算 (基于 FFT 或 DBF)
- [ ] **流并发 (CUDA Streams)**：利用流技术实现数据传输（H2D/D2H）与计算的重叠，进一步降低延迟。
- [ ] **可视化**：接入 OpenGL 渲染 Range-Doppler 热力图。

## 🔧 环境依赖 (Prerequisites)

- **OS**: Windows 10 / 11
- **IDE**: Visual Studio 2019 或 2022
- **Compiler**: MSVC (Visual Studio 自带)
- **CUDA Toolkit**: 11.0 或更高版本 (需安装 Visual Studio Integration)
- **GPU**: 支持 CUDA 的 NVIDIA 显卡

## 🔨 编译与运行 (Build & Run)

#### 1. 克隆仓库

#### 2. 确保你已经安装了 CUDA Toolkit 并配置了环境变量。

#### 3. 打开项目

- 进入项目文件夹。
- 双击打开 .sln 解决方案文件 (例如 CudaRadar.sln)。

#### 4. 配置与编译

- 在 Visual Studio 顶部工具栏，将配置设置为 Release 或 Debug，平台设置为 x64。
- 确保你的开发环境已正确识别 CUDA 版本（右键项目 -> 生成依赖项 -> 生成自定义 -> 勾选 CUDA x.x）。
- 按下 Ctrl + Shift + B 生成解决方案。

#### 5. 运行
- 按下 F5 运行调试，或在输出目录找到 .exe 文件直接运行。

## 📂 数据输入格式 (Input Data)

项目支持读取二进制原始数据文件 (`test_data.bin`)。文件由 **6字节的文件头** 和随后的 **数据体** 组成。

- **字节序 (Endianness)**: Little-Endian (小端序)
- **数据类型**: 16-bit Signed Integer (`int16_t`)

若用户未提供文件，则程序会自动生成一个白噪音数据用于运行。

#### 1. 文件头结构 (File Header)

前 6 个字节定义了雷达数据的维度参数：

| Offset (Byte) | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `NumRX` | int16 | 接收天线数量 |
| 0x02 | `NumChirps` | int16 | 每一帧的 Chirp 数量 |
| 0x04 | `NumSamples` | int16 | 每个 Chirp 的采样点数 |

#### 2. 数据体 (Data Payload)

从第 6 字节开始为雷达回波数据，数据按行优先（Row-Major）方式平铺：

- **数据排列**: `[RX, Chirp, Sample]`
- **总大小**: `NumRX * NumChirps * NumSamples * sizeof(int16)` 字节
- **物理意义**: 实数信号 (Real Signal)

## 🤝 贡献 (Contributing)

欢迎提交 Issue 或 Pull Request！ 如果您有更好的优化策略，欢迎一起交流。

## 📄 许可证 (License)

本项目采用 MIT License 许可证。

