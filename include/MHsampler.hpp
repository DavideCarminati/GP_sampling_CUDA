#ifndef MH_SAMPLER
#define MH_SAMPLER
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include "utils.hpp"
#include "cu_utils.hpp"
/*
// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define CURAND_CHECK(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace Eigen;

struct Distribution
{
    VectorXd mean;
    MatrixXd covariance;
    // std::function<VectorXd(VectorXd, VectorXd, MatrixXd)> pdf;
    // std::function<VectorXd(double, VectorXd, MatrixXd)> sampler;
};

struct Data
{
    VectorXd x_train, y_train;
};

struct MHoptions
{   
    int burnin, max_iterations, store_after;
};*/

MatrixXd MHsampler(Data &data, Distribution &prior, Distribution &likelihood, Distribution &proposal, MHoptions opts);

/*
VectorXd uniform_sampler(curandGenerator_t &gen, int num_samples);
// VectorXd uniform_sampler(curandGenerator_t &gen, cudaStream_t stream, int num_samples);
VectorXd uniform_sampler_double(curandGenerator_t &gen, int num_samples);

VectorXd mvn_sampler(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov);
VectorXd mvn_sampler_double(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov);

VectorXd uni_to_multivariate(const VectorXf &random_samples, const VectorXd &mean, const MatrixXd &cov);
VectorXd uni_to_multivariate_double(const VectorXd &random_samples, const VectorXd &mean, const MatrixXd &cov);
*/

#endif