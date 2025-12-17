#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

const int adcNum = 512;
const int chirpNum = 512;
const int rxNum = 4;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void create_symmetric_hanning_window(std::vector<float>& window, int size);

void read_radar_data(int16_t* data, size_t total_points);

void generate_mock_data_file(const std::string& filename);