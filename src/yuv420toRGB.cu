#include "YUV420ToRGB.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <iostream>

static int defult_img_size = 1920*1920*3/2; // Example size for 1920x1920 YUV420 image
static uint8_t* img_buffer_host = new uint8_t[defult_img_size];
static uint8_t* img_buffer_device = nullptr;

__global__ void yuv420toRGBKernel(uint8_t* yuv420, int width, int height, uint8_t* rgb) {
    // sperate Y, U, and V planes
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels) return;
    uint8_t* y = yuv420 + i;
    uint8_t* u = yuv420 + width * height + (i / width / 2) * (width / 2)  + i % width / 2;
    uint8_t* v = yuv420 + width * height + (width * height / 4) + (i / width / 2) * (width / 2) + i % width / 2;
    int r = *y + 1.402 * (*v - 128);
    int g = *y - 0.344136 * (*u - 128) - 0.714136 * (*v - 128);
    int b = *y + 1.772 * (*u - 128);
    rgb[i * 3] = min(max(r, 0), 255);
    rgb[i * 3 + 1] = min(max(g, 0), 255);
    rgb[i * 3 + 2] = min(max(b, 0), 255);
}

void yuv420toRGBInPlace(uint8_t* yuv420, int width, int height, uint8_t* rgb, cudaStream_t stream) {
    int img_size = width * height * 3 / 2;

    if (img_buffer_host == nullptr || img_size > defult_img_size) {
        if (img_buffer_host) {
            delete[] img_buffer_host;
        }
        img_buffer_host = new uint8_t[img_size];
    }
    if (img_buffer_device == nullptr || img_size > defult_img_size) {
        if (img_buffer_device) {
            CUDA_CHECK(cudaFree(img_buffer_device));
        }
        CUDA_CHECK(cudaMalloc(&img_buffer_device, img_size));
    }
    defult_img_size = img_size;

    memcpy(img_buffer_host, yuv420, img_size);

    // Copy YUV420 data to device
    cudaMemcpy(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int pixels = width * height;
    int numThreads = 256;
    int numBlocks = ceil(pixels / (float)numThreads);
    // Launch the kernel with the specified number of blocks and threads
    yuv420toRGBKernel<<<numBlocks, numThreads, 0 , stream>>>(img_buffer_device, width, height, rgb);
}