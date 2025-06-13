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
    int img_size = InputW * InputH * 3 / 2; // YUV420 的大小

    // 將 yuv image 讀取成 uint8_t
    std::ifstream file(ImagePath, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* yuv_pinned = nullptr; // 用於 pinned memory 的指標
    CUDA_CHECK(cudaMallocHost(&yuv_pinned, img_size * sizeof(uint8_t)));
    file.read(reinterpret_cast<char*>(yuv_pinned), fileSize);
    file.close();

    uint8_t* yuv_buffer_device = nullptr; // 用於 GPU 的 YUV buffer 指標
    // 分配 memory
    CUDA_CHECK(cudaMalloc(&yuv_buffer_device, img_size * sizeof(uint8_t)));

    // 計時
    auto t1 = std::chrono::high_resolution_clock::now();
    // Copy YUV420 data to device
    CUDA_CHECK(cudaMemcpy(yuv_buffer_device, yuv_pinned, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    auto t2 = std::chrono::high_resolution_clock::now();
    double duration_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    std::cout << "pinned copy took " << duration_us << " us." << std::endl;

    // 分配 GPU 記憶體給 RGB buffer (output)
    uint8_t* rgb_buffer_device;
    CUDA_CHECK(cudaMalloc(&rgb_buffer_device, 3 * InputW * InputH * sizeof(uint8_t)));
    // 創建 CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // 計時
    t1 = std::chrono::high_resolution_clock::now();
    // 將 yuv 轉換成 rgb
    yuv420toRGBInPlace(yuv_buffer_device, InputW, InputH, rgb_buffer_device, stream);
    // 等待 GPU 完成處理
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t2 = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
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

    // 以下 CPU 版本
    uint8_t* rgb_buffer = new uint8_t[img_size*2];
    // 計時
    t1 = std::chrono::high_resolution_clock::now();
    // 將 yuv 轉換成 rgb
    yuv420toRGBInPlace_cpu(yuv_pinned, InputW, InputH, rgb_buffer, stream);
    t2 = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    std::cout << "yuv420toRGBInPlace_cpu took " << duration_us << " us." << std::endl;
    // 用 cv::Mat 包裝
    cv::Mat rgb_cpu_image(InputH, InputW, CV_8UC3, rgb_buffer);

    // 顯示圖片
    cv::cvtColor(rgb_cpu_image, rgb_cpu_image, cv::COLOR_RGB2BGR); // OpenCV 使用 BGR 格式
    cv::imshow("RGB Image cpu", rgb_cpu_image);
    cv::waitKey(0);


    // 釋放資源
    CUDA_CHECK(cudaFree(rgb_buffer_device));
    CUDA_CHECK(cudaFree(yuv_buffer_device));
    CUDA_CHECK(cudaFreeHost(yuv_pinned));
    delete[] rgb_buffer;
    cv::destroyAllWindows();
    std::cout << "Program finished successfully." << std::endl;
    return 0;
}
