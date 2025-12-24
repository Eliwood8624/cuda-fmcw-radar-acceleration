#pragma once
#include <vector>
#include <complex>


class GpuRadarProcessor
{
public:
    GpuRadarProcessor(int rxNum, int chirpNum, int sampleNum);
    ~GpuRadarProcessor();
    void process(const std::vector<int16_t>& dataInput, std::vector<std::complex<float>>& dataOutput);

private:
    int m_rxNum;
    int m_chirpNum;
    int m_sampleNum;

    struct Impl;
    std::unique_ptr<Impl> pImpl;

};
