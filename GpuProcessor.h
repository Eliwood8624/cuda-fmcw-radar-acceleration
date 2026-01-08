#pragma once
#include <vector>
#include <complex>


class GpuRadarProcessor
{
public:
    GpuRadarProcessor(int rxNum, int chirpNum, int sampleNum);
    ~GpuRadarProcessor();
    void processAsync(const std::vector<int16_t>& dataInput, std::vector<std::complex<float>>& dataOutput, int frameIdx = 0);
    void collectResult(std::vector<std::complex<float>>& out, int frameIdx);
    int getStreamNum() { return STREAM_NUM; };

private:
    int m_rxNum;
    int m_chirpNum;
    int m_sampleNum;

    struct Impl;
    std::unique_ptr<Impl> pImpl;

    static const int STREAM_NUM = 3;
    void initImpl();
};
