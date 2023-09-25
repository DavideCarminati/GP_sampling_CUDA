cmake_minimum_required(VERSION 3.5.0)
project(GPSamplingCUDA VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.2/bin/nvcc)

add_compile_options(-DCMAKE_BUILD_TYPE=Release -DEIGEN_NO_DEBUG)
set(CMAKE_BUILD_TYPE Release)

include(CTest)
enable_testing()

find_package(Eigen3 3.4 REQUIRED)
find_package(Python3 REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(include/
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    ${PYTHON_INCLUDE_DIRS})
add_executable(GPSamplingCUDA src/main.cpp src/MHsampler.cu)



target_link_libraries(  GPSamplingCUDA
                        Eigen3::Eigen
                        # cuda
                        curand
                        cublas
                        cusolver
                        ${PYTHON_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
