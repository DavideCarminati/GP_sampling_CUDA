#include "SMC2sampler.hpp"


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
                                double *w_x_particles_new)
{
    // Accept or reject new time series using MH
    double u = curand_uniform_double(state);
    if (*mlh_new / *mlh >= u)
    {
        theta[0] = theta_new[0];
        theta[1] = theta_new[1];
        *x_theta = *x_theta_new;
        *mlh = *mlh_new;
        *x_particles = *x_particles_new;
        *w_x_particles = *w_x_particles_new;
    }
}
/**
 * Particle rejuvination
*/
__global__
void MarginalMetropolisHastings(curandState_t *state, 
                                const int T_current,
                                double *theta, 
                                double *x, 
                                double *mlh, 
                                double *x_particles, 
                                double *w_x_particles, 
                                const cuData &data)
{
    // BUG: x_theta is a N_theta x N matrix!
    // BUG: mlh is a N_theta vector!
    // BUG: x_particles is a N_x x N_theta matrix!
    // BUG: w_x_particles is a N_x x N_theta matrix!
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Map<MatrixXd> theta_mat(theta, data.data.N_theta, 2);
    Map<MatrixXd> x_mat(x, data.data.N_theta, data.data.N);
    Map<MatrixXd> x_particles_mat(x_particles, data.data.N_x, data.data.N_theta);
    Map<MatrixXd> w_x_particles_mat(w_x_particles, data.data.N_x, data.data.N_theta);
    if (tid < data.data.N_theta)
    {
        double *theta_new = new double[2];
        theta_new[0] = curand_normal_double(state);
        theta_new[1] = curand_normal_double(state);
        double *mlh_new;
        double *x_theta_new, *x_particles_new, *w_x_particles_new;
        cudaMalloc((void**)&x_theta_new, sizeof(double) * data.data.N);
        cudaMalloc((void**)&x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&mlh_new, sizeof(double));
        __syncthreads();
        ParticleFilterPMMH<<<1, 1>>>(theta_new, T_current, data, state, mlh_new, x_theta_new, x_particles_new, w_x_particles_new);
        MetropolisHastingsReject<<<1, 1, 0, cudaStreamTailLaunch>>>
                        (state, theta_mat.row(tid).data(), theta_new, x_mat.row(tid).data(), x_theta_new, &mlh[tid], mlh_new, 
                        x_particles_mat.col(tid).data(), x_particles_new, w_x_particles_mat.col(tid).data(), w_x_particles_new);
        
        // // TODO: Write a kernel that accept/reject new samples and put it in cudaStreamTailLaunch stream. Then return.
        // double u = curand_uniform_double(state);
        // if (*mlh_new / *mlh >= u)
        // {
        //     theta[2 * tid] = theta_new[0];
        //     theta[2 * tid + 1] = theta_new[1];
        //     *x_theta = *x_theta_new;
        //     *mlh = *mlh_new;
        //     *x_particles = *x_particles_new;
        //     *w_x_particles = *w_x_particles_new;
        // }
        // // Until here <---
        
    }


}

__global__
void PropagateState(curandState_t *state, double *x_t, double *w_x_t, const double *L_t, const cuData &data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Map<VectorXd> x_t_vec(x_t, data.data.N_x);
    Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
    Map<const::VectorXd> L_t_vec(L_t, data.data.N); // Check this
    if (tid < data.data.N_x)
    {
        VectorXd rand_var(data.data.N);
        for (int ii = 0; ii < data.data.N; ii++)
        {
            rand_var(ii) = curand_normal_double(&(state[tid]));
        }
        x_t_vec(tid) = L_t_vec.transpose() * rand_var;
        w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[tid] - x_t_vec(tid), 2) / data.data.Rnoise );
    }
}

/**
 * Metropolis Resempling algorithm for creating ancestors
*/
__global__
void MetropolisResampling(curandState_t *global_state, double *weights, const int N_theta/*or N_x if inside the PF*/, const int iters, int* ancestors)
{

    Map<VectorXi> ancestors_vec(ancestors, N_theta);
    printf("N_theta is %d\niters is %d\n", N_theta, iters);
    ancestors_vec.resize(N_theta);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N_theta)
    {
        int k = tid;
        for (int t = 0; t < iters; t++)
        {
            double u = curand_uniform_double(&(global_state[tid]));
            double jd = curand_uniform_double(&(global_state[tid]));
            jd *= (N_theta - 1 + 0.999999);
            int j = (int)trunc(jd);
            if ( u <= weights[j] / weights[k] && !isnan(weights[j] / weights[k]) )
            {
                k = j;
            }
            ancestors_vec(tid) = k;

        }
    }
}

__global__
void Resample(const cuData &data, double *x_t, double *w_x_t, const int* a)
{
    // x and w_x are N_x x 1 vectors since they are referred to the current time instant
    printf("a contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
    double *x_t_old, *w_x_t_old;
    cudaMalloc((void**)&x_t_old, sizeof(double) * data.data.N_x);
    cudaMalloc((void**)&w_x_t_old, sizeof(double) * data.data.N_x);
    memcpy(x_t_old, x_t, data.data.N_x);
    memcpy(w_x_t_old, w_x_t, data.data.N_x);
    // std::copy(x_t, x_t + data.data.N_x, x_t_old);
    // std::copy(w_x_t, w_x_t + data.data.N_x, w_x_t_old);
    // Oppure parallelizzando: if (tid < N_x)...
    for (int ii = 0; ii < data.data.N_x; ii++)
    {
        x_t[ii] = x_t_old[a[ii]];
        w_x_t[ii] = w_x_t_old[a[ii]];
        printf("here %d", ii);
    }
}

__global__
void FinalizePFPMMH(const cuData &data,
                    const double *x,
                    const double *w_x,
                    double *mlh_hat, 
                    double *x_hat, 
                    double *x_particles, 
                    double *w_x_particles)
{
    // Average the states over the weights and return time series, 
    Map<const::MatrixXd> x_mat(x, data.data.N_x, data.data.N);
    Map<const::MatrixXd> w_x_mat(w_x, data.data.N_x, data.data.N);
    // Map<const::VectorXd> mlh_hat_vec(mlh_hat, data.data.N_theta);
    *mlh_hat = (w_x_mat.colwise().sum() / data.data.N_x).prod();
    Map<VectorXd> x_hat_vec(x_hat, data.data.N);
    Map<VectorXd> x_particles_vec(x_particles, data.data.N_x);
    Map<VectorXd> w_x_particles_vec(w_x_particles, data.data.N_x);
    VectorXd w_x_summed = w_x_mat.colwise().sum();
    MatrixXd w_hat_normalized = w_x_mat * w_x_summed.asDiagonal();
    x_hat_vec = ( w_hat_normalized.array() * x_mat.array() ).colwise().sum();
    // x_hat_vec = ( w_x.array() / w_x.colwise().sum().array() * x.array() ).colwise().sum();
    x_particles_vec = x_mat.rightCols(1);
    w_x_particles_vec = w_x_mat.rightCols(1);
}

/**
 * Particle Filter Metropolis Hastings (PMMH)
*/
__global__
void ParticleFilterPMMH(double *theta, 
                        int T_current, 
                        const cuData &data, 
                        curandState_t *global_state, 
                        double *mlh_hat, 
                        double *x_hat, 
                        double *x_particles, 
                        double *w_x_particles)
{
    // (Re)Initialize
    MatrixXd w_x = MatrixXd::Zero(data.data.N_x, T_current + 1);
    w_x.col(0) = VectorXd::Ones(data.data.N_x) / data.data.N_x;
    Map<VectorXd> system_x(data.data.X, data.data.N);

    MatrixXd x = MatrixXd::Zero(data.data.N_x, T_current + 1);
    // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);
    MatrixXd K = computeKernel(system_x, system_x, theta[0], theta[1]);
    // MatrixXd L = K.llt().matrixL();
    MatrixXd L = MatrixXd::Zero(data.data.N, data.data.N);
    cuCholesky(K.data(), data.data.N, L.data());
    printf("L is:\n %f\n%f\t%f\n", L(0,0), L(1,0), L(1,1));
    VectorXd rand_var(data.data.N_x);
    for (int ii = 0; ii < data.data.N_x; ii++)
    {
        rand_var(ii) = curand_normal_double(global_state) + L(0,0) * curand_normal_double(global_state);
    }
    x.col(0) = rand_var;
    // curandState_t local_state = global_state[threadIdx.x];
    VectorXi a(data.data.N_x);

    for (int t = 1; t < T_current + 1; t++)
    {
        // TODO: Write a kernel that initializes these variables to be passed to PropagateState kernel
        // Not necessary if I pass directly eigen matrices with .data()
        // double *L_t, *x_t, *w_x_t;
        // cudaMalloc((void**)&L_t, sizeof(double) * data.data.N);
        // cudaMalloc((void**)&x_t, sizeof(double) * data.data.N_x);
        // cudaMalloc((void**)&w_x_t, sizeof(double) * data.data.N_x);
        // int *a;
        // cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        // Map<VectorXd> L_t_vec(L_t, data.data.N);
        // Map<VectorXd> x_t_vec(x_t, data.data.N_x);
        // Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
        // L_t_vec = L.row(t).transpose();
        // x_t_vec = x.col(t);
        // w_x_t_vec = w_x.col(t);
        // Until here <----

        printf("propagate\n");
        // PropagateState<<<1, 1>>>(global_state, x_t, w_x_t, L_t, data);
        PropagateState<<<1, 1>>>(global_state, x.col(t).data(), w_x.col(t).data(), L.row(t).data(), data);
        
        // Metropolis resampling (__device__ kernel)
        
        
        printf("resample\n");
        // MetropolisResampling<<<1, 1>>>(global_state, w_x_t, data.data.N_x, data.data.B, a);
        MetropolisResampling<<<1, 1>>>(global_state, w_x.col(t).data(), data.data.N_x, data.data.B, a.data());
        
        Resample<<<1, 1>>>(data, x.col(t).data(), w_x.col(t).data(), a.data());

        // TODO: Write a kernel that shuffles x using a. Then cycle repeats.
        // printf("a contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
        // for (int ii = 0; ii < data.data.N_x; ii++)
        // {
        //     x(ii, t) = x_t[a[ii]];
        //     w_x(ii, t) = w_x_t[a[ii]];
        //     printf("here %d", ii);
        // }
        // Until here <---
    }
    // TODO: Write a kernel for cudaStreamTailLaunch stream to finalize the PF, then return.

    FinalizePFPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(data, x.data(), w_x.data(), mlh_hat, x_hat, x_particles, w_x_particles);

    // *mlh_hat = (w_x.colwise().sum() / data.data.N_x).prod();
    // Map<VectorXd> x_hat_vec(x_hat, data.data.N);
    // Map<VectorXd> x_particles_vec(x_particles, data.data.N_x);
    // Map<VectorXd> w_x_particles_vec(w_x_particles, data.data.N_x);
    // VectorXd w_x_summed = w_x.colwise().sum();
    // MatrixXd w_hat_normalized = w_x * w_x_summed.asDiagonal();
    // x_hat_vec = ( w_hat_normalized.array() * x.array() ).colwise().sum();
    // // x_hat_vec = ( w_x.array() / w_x.colwise().sum().array() * x.array() ).colwise().sum();
    // x_particles_vec = x.rightCols(1); // memcheck error
    // w_x_particles_vec = w_x.rightCols(1);

}

/**
 * Bootstrap Particle Filter for one-step ahead prediction
*/
__global__
void ParticleFilter(double *theta, 
                    int T_next, 
                    const cuData &data, 
                    curandState_t *global_state, 
                    double *mlh_hat, 
                    VectorXd &x_hat, 
                    VectorXd &x_particles, 
                    VectorXd &w_x_particles)
{
    // x_hat has to be a vector with T_next elements
    VectorXd w_x = w_x_particles;
    Map<VectorXd> system_x(data.data.X, data.data.N);

    VectorXd x_hat_new = VectorXd::Zero(data.data.N_x);
    // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);
    MatrixXd K = computeKernel(system_x, system_x, theta[0], theta[1]);
    // MatrixXd L = K.llt().matrixL();
    MatrixXd L = MatrixXd::Zero(data.data.N, data.data.N);
    cuCholesky(K.data(), data.data.N, L.data());
    // curandState_t local_state = global_state[threadIdx.x];

    // Propagate for each particle (__device__ kernel)
    RowVectorXd L_t = L.row(T_next);
    VectorXd x_t = x_particles;
    VectorXd w_x_t = w_x;
    PropagateState<<<1, 1>>>(global_state, x_t.data(), w_x_t.data(), L_t.data(), data);
    // Metropolis resampling (__device__ kernel)
    VectorXi a(data.data.N_x);
    MetropolisResampling<<<1, 1>>>(global_state, w_x_t.data(), data.data.N_x, 5000, a.data());
    VectorXd tmp_x = x_hat_new;
    VectorXd tmp_w_x = w_x;
    for (int ii = 0; ii < data.data.N_x; ii++)
    {
        x_hat_new(ii) = tmp_x(a(ii));
        w_x(ii) = tmp_w_x(a(ii));
    }
    // x_hat_new = x_hat_new(a);
    // w_x = w_x(a);

    *mlh_hat = *mlh_hat / data.data.N_x * w_x.sum();
    x_hat(T_next) = (w_x.array() * x_hat_new.array() / w_x.sum()).sum();
    x_particles = x_hat_new;
    w_x_particles = w_x;

}

/**
 * Initialize needed quantities
*/
__global__
void SMC2_init( curandState *state, 
                const cuData &data, 
                double *theta, 
                double *w_theta, 
                double *f, 
                double *mlh, 
                double *f_particles,    // N_x f-particles for each theta-particle
                double *w_f)            // Weights of the N_x f-particles
{
    // Initialize. For each N_theta theta, do the following:
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, 0, 0, &state[tid]);
    Map<MatrixXd> system_x(data.data.X, data.data.N, 1);
    Map<MatrixXd> f_mat(f, data.data.N_theta, data.data.N);
    Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
    Map<MatrixXd> theta_mat(theta, data.data.N_theta, 2);
    Map<MatrixXd> w_theta_mat(w_theta, data.data.N_theta, data.data.N);
    Map<MatrixXd> f_particles_mat(f_particles, data.data.N_x, data.data.N_theta);
    Map<MatrixXd> w_f_mat(w_f, data.data.N_x, data.data.N_theta);
    mlh_vec.setOnes();
    if (tid < data.data.N_theta)
    {
        // Sample N_theta theta-particles and their weight
        // double *theta = new double[2];
        for (int ii = 0; ii < 2; ii++)
        {
            theta_mat(tid, ii) = curand_normal_double(&state[tid]);
        }
        w_theta_mat.col(0) = VectorXd::Ones(data.data.N_theta) / data.data.N_theta;

        // Sample N_theta x0 for each theta-particle
        MatrixXd K = computeKernel(system_x, system_x, theta_mat(tid, 0), theta_mat(tid, 1));
        // MatrixXd L = K.llt().matrixL();
        MatrixXd L = MatrixXd::Zero(data.data.N, data.data.N);
        cuCholesky(K.data(), data.data.N, L.data());
        VectorXd rand_var(data.data.N);
        for (int kk = 0; kk < data.data.N; kk++)
        {
            rand_var(kk) = curand_normal_double(&state[tid]);
        }
        f_mat(tid, 0) = L(0, 0) * rand_var(0);
        
        // Create the particle ensemble at time=0
        MatrixXd rand_mat(data.data.N_x, data.data.N_theta);
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            for (int kk = 0; kk < data.data.N_theta; kk++)
            {
                rand_mat(ii, kk) = curand_normal_double(&(state[tid]));
            }
        }
        f_particles_mat = f_mat.col(0).transpose() + rand_mat;
        w_f_mat = MatrixXd::Ones(data.data.N_x, data.data.N_theta) / data.data.N_x;
    }
}

/**
 * Sequential Monte Carlo² (SMC²) algorithm
*/
void SMC2(const Data &data)
{
    curandState *devStates;
    int totalThreads = 256;
    CUDA_CHECK(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));
    double *dev_theta, *dev_w_theta, *dev_f, *dev_mlh, *dev_f_particles, *dev_w_f;
    int *dev_ancestors;
    CUDA_CHECK(cudaMalloc((void**)&dev_theta, sizeof(double) * 2 * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_w_theta, sizeof(double) * data.N_theta * data.N));
    CUDA_CHECK(cudaMalloc((void**)&dev_f, sizeof(double) * data.N_theta * data.N));
    CUDA_CHECK(cudaMalloc((void**)&dev_mlh, sizeof(double) * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_f_particles, sizeof(double) * data.N_x * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_w_f, sizeof(double) * data.N_x * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_ancestors, sizeof(int) * data.N_theta));
    cuData *dev_data = new cuData;
    dev_data->data = data;
    CUDA_CHECK(cudaMallocManaged((void**)&(dev_data->data.X), sizeof(double) * data.N));
    std::copy(data.X, data.X + sizeof(double) * data.N, dev_data->data.X);
    CUDA_CHECK(cudaMallocManaged((void**)&(dev_data->data.Y), sizeof(double) * data.N));
    std::copy(data.Y, data.Y + sizeof(double) * data.N, dev_data->data.Y);

    int B = 5000;

    // Initialize
    SMC2_init<<<1, 1>>>(devStates, *dev_data, dev_theta, dev_w_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f);
    // Cycle through time
    for (int t = 0; t < data.N - 1; t++)
    {
        // VectorXi ancestors(data.N_theta);
        MetropolisResampling<<<1, 1>>>(devStates, dev_w_theta, dev_data->data.N_theta, B, dev_ancestors);
        VectorXi a(data.N_theta);
        CUDA_CHECK(cudaMemcpy(a.data(), dev_ancestors, sizeof(int) * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "ancestors:\n" << a << std::endl;
        MarginalMetropolisHastings<<<1, 1>>>(devStates, t, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_f_particles, *dev_data);
    }

}