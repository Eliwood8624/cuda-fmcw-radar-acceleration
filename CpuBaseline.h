#pragma once
#include <vector>
#include <complex>
#include "RadarParams.h"
#include "./fftw/fftw3.h"

void cpu_process_old_way(std::vector<int16_t>& indata, std::vector<std::complex<float>>& outdata);