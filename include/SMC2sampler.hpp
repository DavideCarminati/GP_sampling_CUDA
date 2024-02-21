#ifndef SMC2_SAMPLER
#define SMC2_SAMPLER
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include "utilsSMC.hpp"
#include "cu_utils.hpp"
#include "matplotlibcpp_eigen.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

struct Data
{
    double *X;
    double *Y;
    // double *Y_noiseless;
    int N_x, N_theta, N, B;
    double Rnoise;
};

class cuData : public Managed
{
    public:
        Data data;
};

__global__
void cudaFreePF(double *L, int *a);

__global__
void MetropolisHastingsReject(  curandState_t *state,
                                const cuData &data,
                                double *theta,
                                double *theta_new,
                                double *x_theta,
                                double *x_theta_new,
                                double *mlh,
                                double *mlh_new,
                                double *x_particles,
                                double *x_particles_new,
                                double *w_x_particles,
                                double *w_x_particles_new);

__global__
void FinalizePFPMMH(const cuData &data,
                    double *x,              // [N_x x N] All x-particles for each time instant
                    double *w_x,            // Matrix of N_x weights for all N steps
                    double *mlh_hat,        // [1 x 1] Marginal LH referred to this theta vector
                    double *x_hat,          // [N x 1] Time series for this theta vector
                    double *x_particles,    // [N_x x 1]
                    double *w_x_particles); // Vector of only the last N_x weights

__global__
void PermutateStatesAndWeights(const cuData &data, double *x_t, double *w_x_t, const int* a);

__global__
void MetropolisResampling(curandState_t *global_state, double *weights, const int N_theta, const int iters, int* ancestors);
__global__
void PropagateState(curandState_t *global_state, const int T_current, double *x_t, double *w_x_t, double *L, const cuData &data);
__global__
void MarginalMetropolisHastings(curandState_t *global_state_theta,
                                curandState_t *global_state_x,
                                const int T_current,
                                double *theta,                  // [2 x N_theta]
                                double *x,                      // [N x N_theta]
                                double *mlh,                    // [N_theta x 1]
                                double *x_particles,            // [N_x x N_theta]
                                double *w_x_particles,          // [N_x x N_theta]
                                const cuData &data);
__global__
void ParticleFilterPMMH(double *theta, 
                        int T_current, 
                        const cuData &data, 
                        curandState_t *global_state, 
                        double *mlh_hat, 
                        double *x_hat, 
                        double *x_particles, 
                        double *w_x_particles);
/**
 * Finalize Particle Filter matrices
*/
__global__
void FinalizePF(const cuData &data,
                const int T_next,
                double *x_predicted,            // PF one-step prediction
                double *w_x_predicted,          // PF predicted last-step weights
                double *mlh_hat,                // PF estimated mlh until step t updated to t+1
                double *x_hat_theta,            // Updated trajectory at t+1 for each theta
                double *x_particles,            // N_x particles for next iteration
                double *w_x_particles);

__global__
void ParticleFilter(double *theta,                  // [2 x N_theta] Matrix with all thetas
                    const int T_next, 
                    const cuData &data, 
                    curandState_t *global_state, 
                    double *mlh_hat,                // [N_theta x 1] Marginal LH for each theta 
                    double *x_hat,                  // [N x N_theta] Time series for each theta
                    double *x_particles,            // [N_x x N_theta] N_x particles obtained in the last step
                    double *w_x_particles);

__global__
void SMC2_init( curandState *state, 
                const cuData &data, 
                double *theta, 
                double *w_theta, 
                double *f, 
                double *mlh, 
                double *f_particles, 
                double *w_f);

__global__
void PermutateThetaAndWeights(  const cuData &data, 
                                double *theta,          // [2 x N_theta] matrix of parameters a t=T_current
                                double *x_hat,          // [N x N_theta] State trajectory t=1:T_current
                                double *mlh,            // [N_theta x 1] Marginal LH at current time
                                double *x_particles,    // [N_x x N_theta] N_x particles for each theta
                                double *w_x_particles,  // [N_x x N_theta] Weights
                                const int* a);


__global__
void NormalizeWeights(  const cuData &data,
                        const double *mlh_hat,      // [N_theta x 1] Marginal LH at current time
                        double *w_theta             // [N_theta x 1] (Normalized) theta weights at time T_next
                      );

// __global__
// void ResampleThetaParticles(curandState_t *global_state, 
//                             const int T_current,
//                             double *theta_weights,          // [N_theta x N] matrix
//                             const int N_theta,
//                             const int iters,
//                             int* ancestors);

void SMC2(const Data &data);

#endif