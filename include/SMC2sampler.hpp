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

using namespace Eigen;

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
void MetropolisHastingsReject(  curandState_t *state,
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
                    const double *x,
                    const double *w_x,
                    double *mlh_hat, 
                    double *x_hat, 
                    double *x_particles, 
                    double *w_x_particles);

__global__
void Resample(const cuData &data, double *x_t, double *w_x_t, const int* a);

__global__
void MetropolisResampling(curandState_t *global_state, double *weights, const int N_theta, const int iters, int* ancestors);
__global__
void PropagateState(curandState_t *global_state, const int T_current, double *x_t, double *w_x_t, double *L, const cuData &data);
__global__
void MarginalMetropolisHastings(curandState_t *state, 
                                const int T_current,
                                double *theta, 
                                double *x_theta, 
                                double *mlh, 
                                double *x_particles, 
                                double *w_x_particles, 
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
__global__
void ParticleFilter(double *theta, 
                    int T_current, 
                    const cuData &data, 
                    curandState_t *global_state, 
                    double *mlh_hat, 
                    VectorXd &x_hat, 
                    VectorXd &x_particles, 
                    VectorXd &w_x_particles);

__global__
void SMC2_init( curandState *state, 
                const cuData &data, 
                double *theta, 
                double *w_theta, 
                double *f, 
                double *mlh, 
                double *f_particles, 
                double *w_f);

void SMC2(const Data &data);

#endif