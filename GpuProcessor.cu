#include "GpuProcessor.h"

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
    int16_t* d_inData;
    cufftComplex* d_fft1d_data, * d_fft2d_data, * d_temp_Data;
    float* d_rangeWin, * d_dopplerWin;

    size_t inDataSize = inData.size() * sizeof(int16_t);
    size_t fft1dSize = inData.size() * sizeof(cufftComplex);
    size_t fft2dSize = (adcNum / 2) * chirpNum * rxNum * sizeof(cufftComplex);
    size_t rangeWinSize = adcNum * sizeof(float);
    size_t dopplerWinSize = chirpNum * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_inData, inDataSize));
    CHECK_CUDA(cudaMalloc((void**)&d_fft1d_data, fft1dSize));
    CHECK_CUDA(cudaMalloc((void**)&d_temp_Data, fft1dSize));
    CHECK_CUDA(cudaMalloc((void**)&d_fft2d_data, fft2dSize));
    CHECK_CUDA(cudaMalloc((void**)&d_rangeWin, rangeWinSize));
    CHECK_CUDA(cudaMalloc((void**)&d_dopplerWin, dopplerWinSize));
    CHECK_CUDA(cudaMemcpy(d_inData, inData.data(), inDataSize, cudaMemcpyHostToDevice));

    std::vector<float> h_WinBuf;
    create_symmetric_hanning_window(h_WinBuf, adcNum);
    CHECK_CUDA(cudaMemcpy(d_rangeWin, h_WinBuf.data(), rangeWinSize, cudaMemcpyHostToDevice));
    create_symmetric_hanning_window(h_WinBuf, chirpNum);
    CHECK_CUDA(cudaMemcpy(d_dopplerWin, h_WinBuf.data(), dopplerWinSize, cudaMemcpyHostToDevice));


    // 变量类型转换以及加窗
    int threads = 256;
    int blocks_pre = (rxNum * chirpNum * adcNum + threads - 1) / threads;
    preprocess_range_window_kernel <<<blocks_pre, threads >>> (
        d_inData, d_fft1d_data, d_rangeWin,
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
    if (cufftExecC2C(plan_range, d_fft1d_data, d_fft1d_data, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 加窗转置
    blocks_pre = (rxNum * chirpNum * adcNum / 2 + threads - 1) / threads;
    transpose_discard_doppler_window_kernel <<<blocks_pre, threads >>> (
        d_fft1d_data, d_fft2d_data, d_rangeWin,
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
    if (cufftExecC2C(plan_doppler, d_fft2d_data, d_temp_Data, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 转置
    blocks_pre = (rxNum * chirpNum * adcNum / 2 + threads - 1) / threads;
    transpose_doppler_kernel <<<blocks_pre, threads >>> (
        d_temp_Data, d_fft2d_data,
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
    CHECK_CUDA(cudaMemcpy(outData.data(), d_fft2d_data, fft2dSize, cudaMemcpyDeviceToHost));

    cufftDestroy(plan_range);
    cufftDestroy(plan_doppler);
    cudaFree(d_fft1d_data);
    cudaFree(d_fft2d_data);
    cudaFree(d_inData);
    cudaFree(d_rangeWin);
    cudaFree(d_dopplerWin);
}

