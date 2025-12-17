#include"RadarParams.h"

void create_symmetric_hanning_window(std::vector<float>& window, int size)
{
    if (window.size() != size)
    {
        window.resize(size);
    }

    for (int i = 0; i < size; ++i)
    {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (i + 1.0f) / (size + 1.0f)));
    }
}

void read_radar_data(int16_t* data, size_t total_points)
{
    int16_t readXSize, readYSize, readZSize;

    std::ifstream file("test_data.bin", std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "无法打开文件！" << std::endl;
        return;
    }

    int16_t buffer[3];
    file.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    if (file.gcount() == sizeof(buffer))
    {
        readXSize = buffer[0];
        readYSize = buffer[1];
        readZSize = buffer[2];

        std::cout << "Read: " << readXSize << ", " << readYSize << ", " << readZSize << std::endl;
        if (readXSize * readYSize * readZSize != total_points)
        {
            std::cout << "读取失败，读取长度与文件头长度不匹配 " << std::endl;
        }
        file.read(reinterpret_cast<char*>(data), total_points * sizeof(int16_t));
    }
    else
    {
        std::cerr << "读取失败或文件不足6字节" << std::endl;
    }

    if (file)
    {
        std::cout << "读取成功，共读取 " << file.gcount() << " 字节" << std::endl;
    }
    else
    {
        std::cout << "读取不完整，实际只读了 " << file.gcount() << " 字节" << std::endl;
    }

    file.close();
}

void generate_mock_data_file(const std::string& filename)
{
    std::cout << "正在生成测试数据: " << filename << "..." << std::endl;
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "无法创建文件" << std::endl;
        return;
    }

    // 1. 写入 Header，和 read_radar_data 对应)
    int16_t header[3] = {(int16_t)adcNum, (int16_t)chirpNum, (int16_t)rxNum};
    fwrite(header, sizeof(int16_t), 3, fp);

    // 2. 写入数据体
    size_t total_elements = adcNum * chirpNum * rxNum;
    // 使用 buffer 批量写入以提高速度
    const size_t BATCH_SIZE = 1024 * 1024;
    std::vector<int16_t> buffer(BATCH_SIZE);

    srand((unsigned int)time(NULL));

    size_t remain = total_elements;
    while (remain > 0) {
        size_t current_batch = (remain < BATCH_SIZE) ? remain : BATCH_SIZE;
        for (size_t i = 0; i < current_batch; ++i) {
            buffer[i] = (int16_t)(rand() % 4096 - 2048); // 模拟 ADC 数据
        }
        fwrite(buffer.data(), sizeof(int16_t), current_batch, fp);
        remain -= current_batch;
    }

    fclose(fp);
    std::cout << "数据生成完成，大小匹配: " << adcNum << "x" << chirpNum << "x" << rxNum << std::endl;
}