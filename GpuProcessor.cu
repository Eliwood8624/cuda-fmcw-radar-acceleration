#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
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

class CufftPlan {
public:
    CufftPlan() : plan_(0) {}
    CufftPlan(int nx, cufftType type, int batch) {
        if (cufftPlan1d(&plan_, nx, type, batch) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan1d failed");
        }
    }

    ~CufftPlan() {
        if (plan_) {
            cufftDestroy(plan_);
        }
    }

    void reset(int nx, cufftType type, int batch) {
        if (plan_) {
            cufftDestroy(plan_);
            plan_ = 0;
        }
        if (cufftPlan1d(&plan_, nx, type, batch) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan1d failed during reset");
        }
    }

    // 禁止拷贝
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;

    // 只读
    cufftHandle get() const { return plan_; }

private:
    cufftHandle plan_ = 0;
}; 

struct GpuRadarProcessor::Impl
{
    DevicePtr<int16_t> d_inData;
    DevicePtr<cufftComplex> d_fft1d_data;
    DevicePtr<cufftComplex> d_temp_Data;
    DevicePtr<cufftComplex> d_fft2d_data;
    DevicePtr<float> d_rangeWin;
    DevicePtr<float> d_dopplerWin;
    CufftPlan rangeFftPlan;
    CufftPlan dopplerFftPlan;
};

GpuRadarProcessor::GpuRadarProcessor(int rxNum, int chirpNum, int sampleNum)
{
    m_rxNum = rxNum;
    m_chirpNum = chirpNum;
    m_sampleNum = sampleNum;

    pImpl = std::make_unique<Impl>();

    size_t inDataSize = m_rxNum * m_chirpNum * m_sampleNum;
    size_t fft1dSize = m_rxNum * m_chirpNum * m_sampleNum;
    size_t fft2dSize = (m_sampleNum / 2) * m_chirpNum * m_rxNum;
    size_t rangeWinSize = sampleNum;
    size_t dopplerWinSize = m_chirpNum;
    pImpl->d_inData = make_device_unique<int16_t>(inDataSize);
    pImpl->d_fft1d_data = make_device_unique<cufftComplex>(fft1dSize);
    pImpl->d_temp_Data = make_device_unique<cufftComplex>(fft1dSize);
    pImpl->d_fft2d_data = make_device_unique<cufftComplex>(fft2dSize);
    pImpl->d_rangeWin = make_device_unique<float>(rangeWinSize);
    pImpl->d_dopplerWin = make_device_unique<float>(dopplerWinSize);

    std::vector<float> h_WinBuf;
    create_symmetric_hanning_window(h_WinBuf, sampleNum);
    CHECK_CUDA(cudaMemcpy(pImpl->d_rangeWin.get(), h_WinBuf.data(), rangeWinSize * sizeof(float), cudaMemcpyHostToDevice));
    create_symmetric_hanning_window(h_WinBuf, chirpNum);
    CHECK_CUDA(cudaMemcpy(pImpl->d_dopplerWin.get(), h_WinBuf.data(), dopplerWinSize * sizeof(float), cudaMemcpyHostToDevice));

    int fft1dLoopNum = chirpNum * rxNum;
    int fft2dLoopNum = (adcNum / 2) * rxNum;
    pImpl->rangeFftPlan.reset(sampleNum, CUFFT_C2C, fft1dLoopNum);
    pImpl->dopplerFftPlan.reset(chirpNum, CUFFT_C2C, fft2dLoopNum);
}

GpuRadarProcessor::~GpuRadarProcessor() = default;

// inData[rxNum][chirpNum][adcNum]
void GpuRadarProcessor::process(const std::vector<int16_t>& dataInput, std::vector<std::complex<float>>& dataOutput)
{
    size_t inDataSize = m_rxNum * m_chirpNum * m_sampleNum;
    CHECK_CUDA(cudaMemcpy(pImpl->d_inData.get(), dataInput.data(), inDataSize * sizeof(int16_t), cudaMemcpyHostToDevice));

    // 变量类型转换以及加窗
    int threads = 256;
    int blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum + threads - 1) / threads;
    preprocess_range_window_kernel <<<blocks_pre, threads>>> (
        pImpl->d_inData.get(), pImpl->d_fft1d_data.get(), pImpl->d_rangeWin.get(),
        m_rxNum * m_chirpNum * m_sampleNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());

    // 1dfft
    if (cufftExecC2C(pImpl->rangeFftPlan.get(), pImpl->d_fft1d_data.get(), pImpl->d_fft1d_data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 加窗转置
    blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum / 2 + threads - 1) / threads;
    transpose_discard_doppler_window_kernel <<<blocks_pre, threads>>> (
        pImpl->d_fft1d_data.get(), pImpl->d_fft2d_data.get(), pImpl->d_rangeWin.get(),
        m_rxNum, m_chirpNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());

    //fft2d
    if (cufftExecC2C(pImpl->dopplerFftPlan.get(), pImpl->d_fft2d_data.get(), pImpl->d_temp_Data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 转置
    blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum / 2 + threads - 1) / threads;
    transpose_doppler_kernel <<<blocks_pre, threads>>> (
        pImpl->d_temp_Data.get(), pImpl->d_fft2d_data.get(),
        m_rxNum, m_chirpNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());
    ////调试查看数据
    //{
    //    std::complex<float>* host_debug = new std::complex<float>[100];
    //    cudaMemcpy(host_debug, pImpl->d_fft2d_data.get(), 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(host_debug, pImpl->d_fft2d_data.get() + m_chirpNum, 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    delete[] host_debug;
    //}

    //等待 GPU 完成 (用于计时准确性)
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t fft2dSize = (m_sampleNum / 2) * m_chirpNum * m_rxNum;
    if (dataOutput.size() != fft2dSize)
    {
        dataOutput.resize(fft2dSize);
    }
    CHECK_CUDA(cudaMemcpy(dataOutput.data(), pImpl->d_fft2d_data.get(), fft2dSize * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));
}

