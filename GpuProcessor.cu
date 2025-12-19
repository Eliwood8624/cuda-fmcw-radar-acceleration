#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include "GpuProcessor.h"
#include "RadarParams.h"

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

struct CudaDeleter 
{
    void operator()(void* ptr) const 
    {
        if (ptr) 
        {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) 
            {
                std::cerr << "CUDA Free Error: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

template <typename T>
using DevicePtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
DevicePtr<T> make_device_unique(size_t size)
{
    T* outPtr = nullptr;
    cudaError_t err = cudaMalloc(&outPtr, size * sizeof(T));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaMalloc failed: "));
    }
    return DevicePtr<T>(outPtr);
}

// fft1d前的加窗以及数据转换
__global__ void preprocess_range_window_kernel(
    const int16_t* __restrict__ input,
    cufftComplex* __restrict__ output,
    const float* __restrict__ range_window,
    int total_elements,
    int adc_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        int adcIdx = idx % adc_len;
        output[idx].x = (float)(input[idx] - 2048) * range_window[adcIdx];
        output[idx].y = 0.0f;
    }
}

// fft1d后的截断转置并加窗
// input[rxNum][chirpNum][adcNum]
// output[rxNum][adcNum / 2][chirpNum]
__global__ void transpose_discard_doppler_window_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    const float* __restrict__ doppler_window,
    int rx_num, int chirp_num, int adc_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int adc_half = adc_len / 2;
    int total_elements = rx_num * chirp_num * adc_half;
    if (idx < total_elements)
    {
        int outIdx = idx;
        int outChirpIdx = outIdx % chirp_num;
        int outAdcHalfIdx = (outIdx / chirp_num) % adc_half;
        int outRxIdx = (outIdx / chirp_num) / adc_half;
        int inIdx = outAdcHalfIdx + outChirpIdx * adc_len + outRxIdx * adc_len * chirp_num;
        output[outIdx].x = input[inIdx].x * doppler_window[outChirpIdx];
        output[outIdx].y = input[inIdx].y * doppler_window[outChirpIdx];
    }
}

// fft2d后的转置
// input[rxNum][adcNum / 2][chirpNum]
// output[rxNum][chirpNum][adcNum / 2]
__global__ void transpose_doppler_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    int rx_num, int chirp_num, int adc_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int adc_half = adc_len / 2;
    int total_elements = rx_num * chirp_num * adc_half;
    if (idx < total_elements)
    {
        int outIdx = idx;
        int outAdcHalfIdx = outIdx % adc_half;
        int outChirpIdx = (outIdx / adc_half) % chirp_num;
        int outRxIdx = (outIdx / adc_half) / chirp_num;
        int inIdx = outChirpIdx + outAdcHalfIdx * chirp_num + outRxIdx * adc_half * chirp_num;
        output[outIdx].x = input[inIdx].x;
        output[outIdx].y = input[inIdx].y;
    }
}

// inData[rxNum][chirpNum][adcNum]
void gpu_process_new_way(std::vector<int16_t>& inData, std::vector<std::complex<float>>& outData)
{
    size_t inDataSize = inData.size();
    size_t fft1dSize = inData.size();
    size_t fft2dSize = (adcNum / 2) * chirpNum * rxNum;
    size_t rangeWinSize = adcNum;
    size_t dopplerWinSize = chirpNum;
    auto d_inData = make_device_unique<int16_t>(inDataSize);
    auto d_fft1d_data = make_device_unique<cufftComplex>(fft1dSize);
    auto d_temp_Data = make_device_unique<cufftComplex>(fft1dSize);
    auto d_fft2d_data = make_device_unique<cufftComplex>(fft2dSize);
    auto d_rangeWin = make_device_unique<float>(rangeWinSize);
    auto d_dopplerWin = make_device_unique<float>(dopplerWinSize);
    CHECK_CUDA(cudaMemcpy(d_inData.get(), inData.data(), inDataSize, cudaMemcpyHostToDevice));

    std::vector<float> h_WinBuf;
    create_symmetric_hanning_window(h_WinBuf, adcNum);
    CHECK_CUDA(cudaMemcpy(d_rangeWin.get(), h_WinBuf.data(), rangeWinSize, cudaMemcpyHostToDevice));
    create_symmetric_hanning_window(h_WinBuf, chirpNum);
    CHECK_CUDA(cudaMemcpy(d_dopplerWin.get(), h_WinBuf.data(), dopplerWinSize, cudaMemcpyHostToDevice));


    // 变量类型转换以及加窗
    int threads = 256;
    int blocks_pre = (rxNum * chirpNum * adcNum + threads - 1) / threads;
    preprocess_range_window_kernel <<<blocks_pre, threads >>> (
        d_inData.get(), d_fft1d_data.get(), d_rangeWin.get(),
        rxNum * chirpNum * adcNum, adcNum);
    CHECK_CUDA(cudaGetLastError());

    // 1dfft
    cufftHandle plan_range;
    int fft1dLoopNum = chirpNum * rxNum;
    if (cufftPlan1d(&plan_range, adcNum, CUFFT_C2C, fft1dLoopNum) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Plan creation failed!" << std::endl;
        return;
    }
    if (cufftExecC2C(plan_range, d_fft1d_data.get(), d_fft1d_data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 加窗转置
    blocks_pre = (rxNum * chirpNum * adcNum / 2 + threads - 1) / threads;
    transpose_discard_doppler_window_kernel <<<blocks_pre, threads>>> (
        d_fft1d_data.get(), d_fft2d_data.get(), d_rangeWin.get(),
        rxNum, chirpNum, adcNum);
    CHECK_CUDA(cudaGetLastError());

    //fft2d
    cufftHandle plan_doppler;
    int fft2dLoopNum = (adcNum / 2) * rxNum;
    if (cufftPlan1d(&plan_doppler, chirpNum, CUFFT_C2C, fft2dLoopNum) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Plan creation failed!" << std::endl;
        return;
    }
    if (cufftExecC2C(plan_doppler, d_fft2d_data.get(), d_temp_Data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 转置
    blocks_pre = (rxNum * chirpNum * adcNum / 2 + threads - 1) / threads;
    transpose_doppler_kernel <<<blocks_pre, threads>>> (
        d_temp_Data.get(), d_fft2d_data.get(),
        rxNum, chirpNum, adcNum);
    CHECK_CUDA(cudaGetLastError());
    ////调试查看数据
    //{
    //    std::complex<float>* host_debug = new std::complex<float>[100];
    //    cudaMemcpy(host_debug, d_fft2d_data, 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(host_debug, d_fft2d_data + 512, 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    delete[] host_debug;
    //}

    //等待 GPU 完成 (用于计时准确性)
    CHECK_CUDA(cudaDeviceSynchronize());

    if (outData.size() != fft2dSize)
    {
        outData.resize(fft2dSize);
    }
    CHECK_CUDA(cudaMemcpy(outData.data(), d_fft2d_data.get(), fft2dSize, cudaMemcpyDeviceToHost));

    cufftDestroy(plan_range);
    cufftDestroy(plan_doppler);
}

