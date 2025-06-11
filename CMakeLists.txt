cmake_minimum_required(VERSION 3.12)
project(YUV420ToRGB)

find_package(CUDA REQUIRED)
# Find and include OpenCV
set(OpenCV_DIR "C:/Program Files (x86)/opencv_cuda-4.6.0")
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_language(CUDA)

# === Source files ===
set(COMMON_SOURCES
    src/yuv420toRGB.cu
)
set(COMMON_HEADERS
    src/yuv420toRGB.h
)
# === Test executable ===
set(EXEC_SOURCES
    YUVToRGB.cpp
)
# Add executable
add_executable(${PROJECT_NAME} ${EXEC_SOURCES} ${COMMON_SOURCES} ${COMMON_HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
