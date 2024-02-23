cmake_minimum_required(VERSION 3.18.0)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")
project(GPSamplingCUDA VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
# add_compile_options(-DCMAKE_BUILD_TYPE=Release -DEIGEN_NO_DEBUG -fopenmp)
set(CMAKE_BUILD_TYPE Debug)


include(CTest)
enable_testing()

find_package(CUDA)
find_package(Eigen3 3.4 REQUIRED)
find_package(Python3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(OpenMP)

include_directories(include/
                    ${PYTHON_INCLUDE_DIRS}
                    ${CUDA_TOOLKIT_INCLUDE})
# add_executable(GPSamplingCUDA src/main.cpp src/MHsampler.cu src/PTsampler.cu src/SMC2sampler.cu src/utils.cu)
add_executable(GPSamplingCUDA src/mainSMC.cpp src/SMC2sampler.cu src/utilsSMC.cu)
# add_executable(GPSamplingCUDA src/mainSMC.cpp src/SMC2sampler_MT.cu src/utilsSMC_MT.cu)

set_property(TARGET GPSamplingCUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# https://gitlab.kitware.com/cmake/cmake/-/issues/18265
# https://cmake.org/cmake/help/git-master/variable/CMAKE_CUDA_ARCHITECTURES.html
# https://cmake.org/cmake/help/git-master/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(GPSamplingCUDA PROPERTIES CUDA_ARCHITECTURES "61")      # Needed for device linking!
target_compile_options(GPSamplingCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                     #   --generate-line-info # for profiling
                    #    --use_fast_math
                       --relocatable-device-code=true
                     -gencode arch=compute_61,code=sm_61
                       -diag-suppress 20012 # Suppressing Eigen warnings
                       -DPLOT=0
                       -g  # Debug
                        -G # Debug
                       >)
target_link_options(GPSamplingCUDA PRIVATE $<DEVICE_LINK:-arch=sm_61>)

target_link_libraries(  GPSamplingCUDA
                        Eigen3::Eigen
                        # cuda
                        curand
                        cublas
                        cusolver
                        cudadevrt
                        ${PYTHON_LIBRARIES}
                        OpenMP::OpenMP_CXX)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
