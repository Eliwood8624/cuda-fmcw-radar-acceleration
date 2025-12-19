#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "RadarParams.h"
#include "CpuBaseline.h"
#include "GpuProcessor.h"

int main()
{
    size_t total_elements = adcNum * chirpNum * rxNum;
    std::cout << "Data Size: " << adcNum << " x " << chirpNum << " x " << rxNum << " (" << (double)total_elements / 1024 / 1024 << " M points)" << std::endl;
    std::cout << "Processing..." << std::endl;

    // ==========================================
    // --- 数据生成与读取 ---
    // ==========================================
    std::ifstream check_file("test_data.bin");
    if (!check_file.good()) {
        generate_mock_data_file("test_data.bin");
    }
    check_file.close();

    // 设定一个足以击穿L3缓存的内存大小，存储从文件中读取的数据
    const size_t FRAME_SIZE = sizeof(int16_t) * total_elements;
    const size_t TARGET_TOTAL_SIZE = 200 * 1024 * 1024;
    const int NUM_FRAMES = (TARGET_TOTAL_SIZE + FRAME_SIZE - 1) / FRAME_SIZE;
    std::vector<std::vector<int16_t>> all_frames(NUM_FRAMES);    // 初始化数据
    for (int i = 0; i < NUM_FRAMES; i++)
    {
        all_frames[i].resize(total_elements);
        std::cout << "读取第 " << i + 1 << " 帧数据......" << std::endl;
        read_radar_data(all_frames[i].data(), total_elements);
    }

    std::vector<std::complex<float>> cpu_result;
    std::vector<std::complex<float>> gpu_result(total_elements);

    // ==========================================
    // --- CPU 计时 ---
    // ==========================================
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_FRAMES; i++)
    {
        cpu_process_old_way(all_frames[i], cpu_result);
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "[CPU (Simulated FFT + Real Transpose)]" << std::endl;
    std::cout << "Time: " << cpu_ms.count() << " ms" << std::endl;

    // ==========================================
    // --- GPU 预热  ---
    // 通过运行一次，把驱动初始化、Plan创建等耗时操作都做完，时间不计入最终耗时
    // ==========================================
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Warming up GPU (Initializing Context & Planning)..." << std::endl;
    gpu_process_new_way(all_frames[0], gpu_result);

    // 确保预热彻底完成
    cudaDeviceSynchronize();
    std::cout << "Warm-up done. Starting benchmark..." << std::endl;

    // ==========================================
    // --- GPU 计时 ---
    // ==========================================
    auto start_gpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_FRAMES; i++)
    {
        gpu_process_new_way(all_frames[i], gpu_result);
    }

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_ms = end_gpu - start_gpu;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "[GPU (RTX 5060 Real Run)]     Time: " << gpu_ms.count() << " ms" << std::endl;
    std::cout << "Method: CUDA memcpy -> cuFFT 2D -> CUDA memcpy" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 输出加速比
    std::cout << ">>> Speedup: " << cpu_ms.count() / gpu_ms.count() << "x <<<" << std::endl;

    return 0;
}