cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ray_tracing LANGUAGES CUDA)

find_package(PNG REQUIRED)
# sudo apt install -y libpng-dev

include_directories(include)

set(EXEC_PROGRAM "ray_tracing")

add_executable(${EXEC_PROGRAM}
    main.cu
    include/png_writer.h
)

target_link_libraries(${EXEC_PROGRAM} ${PNG_LIBRARY})
