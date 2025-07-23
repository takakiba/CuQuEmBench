cmake_minimum_required(VERSION 3.13.0)

project(custatevec_example LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# for CUDA11 and < CUDA12.8
set(CMAKE_CUDA_ARCHITECTURES 80;80-real;86;86-real;89;89-real;90;90-real)

set(EXECUTABLE qpe.out)

find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(yaml-cpp REQUIRED)

set(SRCROOT "src")

add_executable(
    ${EXECUTABLE}
    ${SRCROOT}/qpe.cu
)

target_include_directories(
    ${EXECUTABLE}
    PUBLIC
    $ENV{CUQUANTUM_ROOT}/include
)

target_link_directories(
    ${EXECUTABLE}
    PUBLIC
    $ENV{CUQUANTUM_ROOT}/lib
)

target_link_libraries(
    ${EXECUTABLE}
    PUBLIC
    custatevec
    HDF5::HDF5
    hdf5::hdf5_cpp
    yaml-cpp
)


