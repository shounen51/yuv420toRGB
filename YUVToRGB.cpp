#include "cuda_utils.h"
#include "yuv420toRGB.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <fstream> // add this line

// usage: ./yolov11_test.exe <image_path> <input_w> <input_h>
int main(int argc, char** argv) {
    // 處理輸入參數
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <input_w> <input_h>" << std::endl;
        return -1;
    }
    const char* image_path = argv[1];
    const int input_w = std::stoi(argv[2]);
    const int input_h = std::stoi(argv[3]);

    // 將 yuv image 讀取成 uint8_t
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* frame = new uint8_t[fileSize];
    file.read(reinterpret_cast<char*>(frame), fileSize);
    file.close();

    // 分配 GPU 記憶體給 RGB buffer
    uint8_t* gpu_rgb_buffer;
    CUDA_CHECK(cudaMalloc(&gpu_rgb_buffer, 3 * input_w * input_h * sizeof(uint8_t)));
    // 創建 CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // 將 yuv 轉換成 rgb
    yuv420toRGBInPlace(frame, input_w, input_h, gpu_rgb_buffer, stream);
    // 等待 GPU 完成處理
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // 釋放 CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 分配 CPU 記憶體回收 GPU 結果
    std::vector<uint8_t> rgb_host(input_w * input_h * 3);
    // 從 GPU 複製回 host
    CUDA_CHECK(cudaMemcpy(rgb_host.data(), gpu_rgb_buffer, input_w * input_h * 3, cudaMemcpyDeviceToHost));
    // 用 cv::Mat 包裝
    cv::Mat rgb_image(input_h, input_w, CV_8UC3, rgb_host.data());

    // 顯示圖片
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR); // OpenCV 使用 BGR 格式
    cv::imshow("RGB Image", rgb_image);
    cv::waitKey(0);
    // 釋放資源
    CUDA_CHECK(cudaFree(gpu_rgb_buffer));
    delete[] frame;
    cv::destroyAllWindows();
    cout << "Program finished successfully." << std::endl;
    return 0;

}
