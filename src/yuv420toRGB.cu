#include "YUV420ToRGB.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <iostream>

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
    // Launch kernel
    int pixels = width * height;
    int numThreads = 256;
    int numBlocks = ceil(pixels / (float)numThreads);
    // Launch the kernel with the specified number of blocks and threads
    yuv420toRGBKernel<<<numBlocks, numThreads, 0 , stream>>>(yuv420, width, height, rgb);
}

void yuv420toRGBCPU(uint8_t* yuv420, int width, int height, uint8_t* rgb) {
    int frameSize = width * height;
    int chromaWidth = width / 2;
    int chromaHeight = height / 2;
    uint8_t* yPlane = yuv420;
    uint8_t* uPlane = yuv420 + frameSize;
    uint8_t* vPlane = yuv420 + frameSize + (frameSize / 4);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int yIdx = y * width + x;
            int uIdx = (y / 2) * chromaWidth + (x / 2);
            int vIdx = (y / 2) * chromaWidth + (x / 2);

            int Y = yPlane[yIdx];
            int U = uPlane[uIdx];
            int V = vPlane[vIdx];

            int r = Y + 1.402 * (V - 128);
            int g = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128);
            int b = Y + 1.772 * (U - 128);

            rgb[yIdx * 3 + 0] = std::min(std::max(r, 0), 255);
            rgb[yIdx * 3 + 1] = std::min(std::max(g, 0), 255);
            rgb[yIdx * 3 + 2] = std::min(std::max(b, 0), 255);
        }
    }
}

void yuv420toRGBInPlace_cpu(uint8_t* yuv420, int width, int height, uint8_t* rgb, cudaStream_t stream) {
    // Launch the kernel with the specified number of blocks and threads
    yuv420toRGBCPU(yuv420, width, height, rgb);
}
