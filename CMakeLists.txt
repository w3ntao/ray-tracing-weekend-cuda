cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# needed by CLion

project(ray_tracing LANGUAGES CUDA)

find_package(PNG QUIET)
if (PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIRS})
    add_definitions(${PNG_DEFINITIONS})
    set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})
else ()
    find_package(PNG)
    message(FATAL_ERROR "libpng-dev is required to build the project.\nPlease install libpng-dev.")
endif ()

include_directories(include)

set(EXEC_PROGRAM "ray_tracing")

add_executable(${EXEC_PROGRAM}
    main.cu
    include/png_writer.h
)

target_link_libraries(${EXEC_PROGRAM} ${PNG_LIBRARY})
