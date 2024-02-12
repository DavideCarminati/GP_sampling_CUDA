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
MatrixXd computeKernel( Eigen::MatrixXd x1,
                        Eigen::MatrixXd x2, 
                        const double amplitude, 
                        const double length_scale
                        );
__device__
void cuCholesky(const double *A, const int lda, double *L);

__device__
void threadBlockDeviceSynchronize(void);
/*
VectorXd uniform_sampler(curandGenerator_t &gen, int num_samples);
// VectorXd uniform_sampler(curandGenerator_t &gen, cudaStream_t stream, int num_samples);
VectorXd uniform_sampler_double(curandGenerator_t &gen, int num_samples);

VectorXi uniform_integer_sampler(curandGenerator_t &gen, int num_samples);
VectorXi uniform_integer_sampler_double(curandGenerator_t &gen, int num_samples);

VectorXd mvn_sampler(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov);
VectorXd mvn_sampler_double(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov);

VectorXd uni_to_multivariate(const VectorXf &random_samples, const VectorXd &mean, const MatrixXd &cov);
VectorXd uni_to_multivariate_double(const VectorXd &random_samples, const VectorXd &mean, const MatrixXd &cov);

__global__ 
void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int,  unsigned int *result);

__global__ 
void setup_kernel(curandState *state);
*/

#endif