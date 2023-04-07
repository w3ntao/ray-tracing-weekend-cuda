cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# required by CLion
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cuda_ray_tracer LANGUAGES CUDA CXX)

set(EXEC_PROGRAM "cudart")

find_package(PNG QUIET)
if (PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIRS})
    add_definitions(${PNG_DEFINITIONS})
    set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})
else ()
    find_package(PNG)
    message(FATAL_ERROR "libpng-dev is required to build the project.\nPlease install libpng-dev.")
endif ()

add_executable(${EXEC_PROGRAM}
        src/main.cu
        src/base/color.cuh
        src/base/image.cu
        src/base/image.cuh
        src/base/util.h
        src/base/ray.cuh
        src/base/vector3.cuh
        src/base/point3.cuh
        src/base/shape.h
        src/base/sphere.h)

target_include_directories(${EXEC_PROGRAM} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${EXEC_PROGRAM} PRIVATE ${PNG_LIBRARY})
