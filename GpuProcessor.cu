#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "GpuProcessor.h"
#include "RadarParams.h"

#include <nvtx3/nvToolsExt.h>

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

struct HostPinnedDeleter
{
    void operator()(void* ptr) const
    {
        if (ptr)
        {
            cudaError_t err = cudaFreeHost(ptr);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA Host Free Error: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

template <typename T>
using HostPinnedPtr = std::unique_ptr<T, HostPinnedDeleter>;

template <typename T>
HostPinnedPtr<T> make_hostPinn_unique(size_t size)
{
    T* outPtr = nullptr;
    cudaError_t err = cudaMallocHost(&outPtr, size * sizeof(T));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaMallocHost failed: "));
    }
    return HostPinnedPtr<T>(outPtr);
}

struct CudaStreamDeleter
{
    void operator()(cudaStream_t s) const
    {
        if (s)
        {
            cudaError_t err = cudaStreamDestroy(s);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA Stream Destroy Error: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

using CudaStream = std::unique_ptr<CUstream_st, CudaStreamDeleter>;

class CufftPlan
{
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

    CufftPlan(const CufftPlan&) = delete;

    CufftPlan& operator=(const CufftPlan&) = delete;

    CufftPlan(CufftPlan&& other) noexcept : plan_(other.plan_) {
        other.plan_ = 0;
    }

    CufftPlan& operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            if (plan_) cufftDestroy(plan_);
            plan_ = other.plan_;
            other.plan_ = 0;
        }
        return *this;
    }

    cufftHandle get() const { return plan_; }
    void reset(int nx, cufftType type, int batch) {
        if (plan_) {
            cufftDestroy(plan_);
            plan_ = 0;
        }
        if (cufftPlan1d(&plan_, nx, type, batch) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan1d failed during reset");
        }
    }


private:
    cufftHandle plan_ = 0;
}; 

struct streamContext
{
    HostPinnedPtr<int16_t> h_pinnedInData;
    HostPinnedPtr<cufftComplex> h_pinnedOutData;
    DevicePtr<int16_t> d_inData;
    DevicePtr<cufftComplex> d_fft1d_data;
    DevicePtr<cufftComplex> d_temp_Data;
    DevicePtr<cufftComplex> d_fft2d_data;
    DevicePtr<float> d_rangeWin;
    DevicePtr<float> d_dopplerWin;
    CufftPlan rangeFftPlan;
    CufftPlan dopplerFftPlan;
};


struct GpuRadarProcessor::Impl
{
    std::vector<CudaStream> streams;
    std::vector<streamContext> sContext;
};

GpuRadarProcessor::GpuRadarProcessor(int rxNum, int chirpNum, int sampleNum)
{
    m_rxNum = rxNum;
    m_chirpNum = chirpNum;
    m_sampleNum = sampleNum;

    initImpl();
}

GpuRadarProcessor::~GpuRadarProcessor() = default;

void GpuRadarProcessor::initImpl()
{
    pImpl = std::make_unique<Impl>();
    pImpl->streams.resize(STREAM_NUM);
    pImpl->sContext.resize(STREAM_NUM);

    for(int i = 0; i < STREAM_NUM; i ++)
    {
        // 初始化stream
        pImpl->streams[i] = CudaStream(nullptr, CudaStreamDeleter());
        cudaStream_t raw_s;
        CHECK_CUDA(cudaStreamCreate(&raw_s));
        pImpl->streams[i].reset(raw_s);

        size_t inDataSize = m_rxNum * m_chirpNum * m_sampleNum;
        size_t fft1dSize = m_rxNum * m_chirpNum * m_sampleNum;
        size_t fft2dSize = (m_sampleNum / 2) * m_chirpNum * m_rxNum;
        size_t rangeWinSize = m_sampleNum;
        size_t dopplerWinSize = m_chirpNum;
        pImpl->sContext[i].h_pinnedInData = make_hostPinn_unique<int16_t>(inDataSize);
        pImpl->sContext[i].h_pinnedOutData = make_hostPinn_unique<cufftComplex>(fft2dSize);
        pImpl->sContext[i].d_inData = make_device_unique<int16_t>(inDataSize);
        pImpl->sContext[i].d_fft1d_data = make_device_unique<cufftComplex>(fft1dSize);
        pImpl->sContext[i].d_temp_Data = make_device_unique<cufftComplex>(fft1dSize);
        pImpl->sContext[i].d_fft2d_data = make_device_unique<cufftComplex>(fft2dSize);
        pImpl->sContext[i].d_rangeWin = make_device_unique<float>(rangeWinSize);
        pImpl->sContext[i].d_dopplerWin = make_device_unique<float>(dopplerWinSize);

        std::vector<float> h_WinBuf;
        create_symmetric_hanning_window(h_WinBuf, m_sampleNum);
        CHECK_CUDA(cudaMemcpy(pImpl->sContext[i].d_rangeWin.get(), h_WinBuf.data(), rangeWinSize * sizeof(float), cudaMemcpyHostToDevice));
        create_symmetric_hanning_window(h_WinBuf, m_chirpNum);
        CHECK_CUDA(cudaMemcpy(pImpl->sContext[i].d_dopplerWin.get(), h_WinBuf.data(), dopplerWinSize * sizeof(float), cudaMemcpyHostToDevice));

        int fft1dLoopNum = m_chirpNum * rxNum;
        int fft2dLoopNum = (adcNum / 2) * rxNum;
        pImpl->sContext[i].rangeFftPlan.reset(m_sampleNum, CUFFT_C2C, fft1dLoopNum);
        pImpl->sContext[i].dopplerFftPlan.reset(m_chirpNum, CUFFT_C2C, fft2dLoopNum);
    }
}

// inData[rxNum][chirpNum][adcNum]
void GpuRadarProcessor::processAsync(const std::vector<int16_t>& dataInput, std::vector<std::complex<float>>& dataOutput, int frameIdx)
{
    int streamIdx = frameIdx % STREAM_NUM;
    cudaStreamSynchronize(pImpl->streams[streamIdx].get());

    size_t inDataSize = m_rxNum * m_chirpNum * m_sampleNum;
    nvtxRangePush("H2P memcpy"); // 在时间轴上开始一个叫 "H2D Copy" 的色块
    memcpy(pImpl->sContext[streamIdx].h_pinnedInData.get(), dataInput.data(), inDataSize * sizeof(int16_t));
    nvtxRangePop();            // 结束色块
    CHECK_CUDA(cudaMemcpyAsync(pImpl->sContext[streamIdx].d_inData.get(), pImpl->sContext[streamIdx].h_pinnedInData.get(), inDataSize * sizeof(int16_t), cudaMemcpyHostToDevice, pImpl->streams[streamIdx].get()));

    // 变量类型转换以及加窗
    int threads = 256;
    int blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum + threads - 1) / threads;
    preprocess_range_window_kernel <<<blocks_pre, threads, 0, pImpl->streams[streamIdx].get() >> > (
        pImpl->sContext[streamIdx].d_inData.get(), pImpl->sContext[streamIdx].d_fft1d_data.get(), pImpl->sContext[streamIdx].d_rangeWin.get(),
        m_rxNum * m_chirpNum * m_sampleNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());

    // 1dfft
    cufftSetStream(pImpl->sContext[streamIdx].rangeFftPlan.get(), pImpl->streams[streamIdx].get());
    if (cufftExecC2C(pImpl->sContext[streamIdx].rangeFftPlan.get(), pImpl->sContext[streamIdx].d_fft1d_data.get(), pImpl->sContext[streamIdx].d_fft1d_data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 加窗转置
    blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum / 2 + threads - 1) / threads;
    transpose_discard_doppler_window_kernel <<<blocks_pre, threads, 0, pImpl->streams[streamIdx].get()>>> (
        pImpl->sContext[streamIdx].d_fft1d_data.get(), pImpl->sContext[streamIdx].d_fft2d_data.get(), pImpl->sContext[streamIdx].d_dopplerWin.get(),
        m_rxNum, m_chirpNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());

    //fft2d
    cufftSetStream(pImpl->sContext[streamIdx].dopplerFftPlan.get(), pImpl->streams[streamIdx].get());
    if (cufftExecC2C(pImpl->sContext[streamIdx].dopplerFftPlan.get(), pImpl->sContext[streamIdx].d_fft2d_data.get(), pImpl->sContext[streamIdx].d_temp_Data.get(), CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        std::cerr << "CUFFT Exec failed!" << std::endl;
        return;
    }

    // 转置
    blocks_pre = (m_rxNum * m_chirpNum * m_sampleNum / 2 + threads - 1) / threads;
    transpose_doppler_kernel <<<blocks_pre, threads, 0, pImpl->streams[streamIdx].get()>>> (
        pImpl->sContext[streamIdx].d_temp_Data.get(), pImpl->sContext[streamIdx].d_fft2d_data.get(),
        m_rxNum, m_chirpNum, m_sampleNum);
    CHECK_CUDA(cudaGetLastError());
    ////调试查看数据
    //{
    //    std::complex<float>* host_debug = new std::complex<float>[100];
    //    cudaMemcpy(host_debug, pImpl->d_fft2d_data.get(), 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(host_debug, pImpl->d_fft2d_data.get() + m_chirpNum, 100 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    //    delete[] host_debug;
    //}

    size_t fft2dSize = (m_sampleNum / 2) * m_chirpNum * m_rxNum;
    if (dataOutput.size() != fft2dSize)
    {
        dataOutput.resize(fft2dSize);
    }
    CHECK_CUDA(cudaMemcpyAsync(pImpl->sContext[streamIdx].h_pinnedOutData.get(), pImpl->sContext[streamIdx].d_fft2d_data.get(), fft2dSize * sizeof(std::complex<float>), cudaMemcpyDeviceToHost, pImpl->streams[streamIdx].get()));
}

void GpuRadarProcessor::collectResult(std::vector<std::complex<float>>& out, int frameIdx) {
    int idx = frameIdx % STREAM_NUM;
    size_t outSize = (m_sampleNum / 2) * m_chirpNum * m_rxNum;

    nvtxRangePush("P2H memcpy"); // 在时间轴上开始一个叫 "H2D Copy" 的色块
    cudaStreamSynchronize(pImpl->streams[idx].get()); // 确保 DtoH 完成
    memcpy(out.data(), pImpl->sContext[idx].h_pinnedOutData.get(), outSize * sizeof(std::complex<float>)); // 安全搬运
    nvtxRangePop();            // 结束色块
}
