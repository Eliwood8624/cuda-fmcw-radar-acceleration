#pragma once
#include <vector>
#include <complex>
#include "RadarParams.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

void gpu_process_new_way(std::vector<int16_t>& inData, std::vector<std::complex<float>>& outData);
