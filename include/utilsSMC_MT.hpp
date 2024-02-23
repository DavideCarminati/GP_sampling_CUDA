#ifndef UTILS
#define UTILS
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include "cu_utils.hpp"

using namespace Eigen;

__host__ __device__
void computeKernel( const double *x1,
                    const int n,
                    const double *x2, 
                    const int m,
                    const double amplitude, 
                    const double l,
                    double *K
                    );
__device__
void cuCholesky(const double *A, const int lda, double *L);

// __global__ 
// void setup_curand_theta(curandStateMtgp32 *state);

// __global__
// void setup_curand_x(curandStateMtgp32 *state);

__device__
void print_matrix(const int &m, const int &n, const double *A, const int &lda);

#endif