#include "cuda_utils.h"
#include "yuv420toRGB.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <fstream>


// usage: ./yolov11_test.exe <image_path> <input_w> <input_h>
int main(int argc, char** argv) {
    // 處理輸入參數
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <input_w> <input_h>" << std::endl;
        return -1;
    }
    const char* ImagePath = argv[1];
    const int InputW = std::stoi(argv[2]);
    const int InputH = std::stoi(argv[3]);

    // 將 yuv image 讀取成 uint8_t
    std::ifstream file(ImagePath, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* frame = new uint8_t[fileSize];
    file.read(reinterpret_cast<char*>(frame), fileSize);
    file.close();

    int img_size = InputW * InputH * 3 / 2; // YUV420 的大小
    uint8_t* yuv_buffer_host = new uint8_t[img_size];
    uint8_t* yuv_buffer_device = nullptr;
    CUDA_CHECK(cudaMalloc(&yuv_buffer_device, img_size * sizeof(uint8_t)));
    memcpy(yuv_buffer_host, frame, img_size);
    // Copy YUV420 data to device
    cudaMemcpy(yuv_buffer_device, yuv_buffer_host, img_size, cudaMemcpyHostToDevice);

    // 分配 GPU 記憶體給 RGB buffer
    uint8_t* rgb_buffer_device;
    CUDA_CHECK(cudaMalloc(&rgb_buffer_device, 3 * InputW * InputH * sizeof(uint8_t)));
    // 創建 CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // 計時
    auto t1 = std::chrono::high_resolution_clock::now();
    // 將 yuv 轉換成 rgb
    yuv420toRGBInPlace(yuv_buffer_device, InputW, InputH, rgb_buffer_device, stream);
    // 等待 GPU 完成處理
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t2 = std::chrono::high_resolution_clock::now();
    double duration_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    std::cout << "yuv420toRGBInPlace took " << duration_us << " us." << std::endl;
    // 釋放 CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 分配 CPU 記憶體回收 GPU 結果
    std::vector<uint8_t> rgb_host(InputW * InputH * 3);
    // 從 GPU 複製回 host
    CUDA_CHECK(cudaMemcpy(rgb_host.data(), rgb_buffer_device, InputW * InputH * 3, cudaMemcpyDeviceToHost));
    // 用 cv::Mat 包裝
    cv::Mat rgb_image(InputH, InputW, CV_8UC3, rgb_host.data());

    // 顯示圖片
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR); // OpenCV 使用 BGR 格式
    cv::imshow("RGB Image", rgb_image);
    cv::waitKey(0);
    // 釋放資源
    CUDA_CHECK(cudaFree(rgb_buffer_device));
    delete[] frame;
    cv::destroyAllWindows();
    std::cout << "Program finished successfully." << std::endl;
    return 0;

}
