#include "SMC2sampler.hpp"


__global__  // CHILD KERNEL FIRST LEVEL
void MetropolisHastingsReject(  curandState_t *state,
                                const cuData &data,
                                const int T_current,
                                double *theta,
                                double *theta_new,
                                double *x_theta,
                                double *x_theta_new,            // [(T_current + 1)] new sampled time series up to t = T_current
                                double *mlh,
                                double *mlh_new,
                                double *x_particles,
                                double *x_particles_new,
                                double *w_x_particles,
                                double *w_x_particles_new)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        #if VERBOSE
            printf("[MHR] Inside MH reject\n");
        #endif
        curandState local_state = state[tid];
        // Accept or reject new time series using MH
        double u = curand_uniform_double(&local_state);
        if (*mlh_new / *mlh >= u)
        {
            theta[0] = theta_new[0];
            theta[1] = theta_new[1];
            *mlh = *mlh_new;
            // memcpy(x_theta, x_theta_new, data.data.N);
            // memcpy(x_particles, x_particles_new, data.data.N);
            // memcpy(w_x_particles, w_x_particles_new, data.data.N);
            memcpy(x_theta, x_theta_new, sizeof(double) * (T_current + 1));
            memcpy(x_particles, x_particles_new, sizeof(double) * data.data.N_x);
            memcpy(w_x_particles, w_x_particles_new, sizeof(double) * data.data.N_x);
            printf("Accepted.\n");
        }
        state[tid] = local_state;
        cudaFree(theta_new);
        cudaFree(x_theta_new);
        cudaFree(x_particles_new);
        cudaFree(w_x_particles_new);
        cudaFree(mlh_new);
    }
}
/**
 * Particle rejuvination
*/
__global__  // PARENT KERNEL
void MarginalMetropolisHastings(curandState_t *global_state_theta,
                                curandState_t *global_state_x,
                                const int T_current,
                                double *theta,                  // [2 x N_theta]
                                double *x,                      // [N x N_theta]
                                double *mlh,                    // [N_theta x 1]
                                double *x_particles,            // [N_x x N_theta]
                                double *w_x_particles,          // [N_x x N_theta]
                                const cuData &data)
{
    // x, x_particles and w_x_particles have size N but full of zeros. I need to consider only the 
    // first T_current rows/columns. In fact the PF PMMH regenerates samples only up to t = T_current.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_theta)
    {
        #if VERBOSE
        if (tid == 0)
        {
            printf("[MMH] tid: %d\n", tid);
        }
        #endif
        curandState local_state = global_state_theta[tid];
        Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
        Map<MatrixXd> x_mat(x, data.data.N, data.data.N_theta);
        Map<MatrixXd> x_particles_mat(x_particles, data.data.N_x, data.data.N_theta);
        Map<MatrixXd> w_x_particles_mat(w_x_particles, data.data.N_x, data.data.N_theta);
        // cudaStream_t streamMMH;
        // cudaStreamCreateWithFlags(&streamMMH, cudaStreamNonBlocking);
    
        // Sample new candidate theta for the new time series
        double *theta_new = new double[2];
        // skipahead(50, &local_state);
        theta_new[0] = 1.0 + curand_normal_double(&local_state);
        theta_new[1] = 1.0 + curand_normal_double(&local_state);
        global_state_theta[tid] = local_state;
        
        double *mlh_new;
        double *x_theta_new, *x_particles_new, *w_x_particles_new;
        // cudaMalloc((void**)&x_theta_new, sizeof(double) * data.data.N);
        cudaMalloc((void**)&x_theta_new, sizeof(double) * (T_current + 1));
        cudaMalloc((void**)&x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&mlh_new, sizeof(double));
        ParticleFilterPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(theta_new, T_current, data, global_state_x, mlh_new, x_theta_new, x_particles_new, w_x_particles_new);
        MetropolisHastingsReject<<<1, 1, 0, cudaStreamTailLaunch>>>
                        (global_state_theta, data, T_current, theta_mat.col(tid).data(), theta_new, x_mat.col(tid).data(), x_theta_new, &mlh[tid], mlh_new, 
                        x_particles_mat.col(tid).data(), x_particles_new, w_x_particles_mat.col(tid).data(), w_x_particles_new);
        // __syncthreads();
        // cudaStreamDestroy(streamMMH);

    }
}

__global__  // CHILD KERNEL SECOND LEVEL
void PropagateState(curandState_t *global_state,
                    const int T_current,            // Time at which prediction is made
                    double *x_t,                    // [N_x x 1] x-particles at T_current
                    double *w_x_t,                  // [N_x x 1] x-weights
                    double *L,                      // [N x N] sqrt(K) st dev matrix
                    const cuData &data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_x)
    {
        #if VERBOSE
        if (1)//(tid == 0)
        {
            printf("[PROP] tid %d. Time %d.\n", tid, T_current);
        }
        #endif
        // printf("N is %d\n", data.data.N);
        // printf("T-th row of L is:\n");
        // printf("L: %f\t%f\n", L[0], L[99]);
        // print_matrix(data.data.N, data.data.N, L, data.data.N);
        // printf("x particles at time %d\n", T_current);
        // print_matrix(data.data.N_x, 1, x_t, data.data.N_x);
        curandState local_state = global_state[tid];
        Map<VectorXd> x_t_vec(x_t, data.data.N_x);
        Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
        Map<MatrixXd> L_mat(L, data.data.N, data.data.N);
    
        VectorXd rand_var(data.data.N);
        // skipahead(100, global_state);
        for (int ii = 0; ii < data.data.N; ii++)
        {
            rand_var(ii) = curand_normal_double(&local_state);
        }
        // x_t_vec(tid) = L_mat.col(T_current).transpose() * rand_var;
        x_t_vec(tid) = L_mat.row(T_current) * rand_var;
        w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[T_current] - x_t_vec(tid), 2) / data.data.Rnoise );
        // printf("[PROP] tid %d. Time: %d. w_x_t is %e; x_t is %f; y is %f\n", tid, T_current, w_x_t[tid], x_t[tid], data.data.Y[T_current]);
        global_state[tid] = local_state;
    }
}

/**
 * Metropolis Resempling algorithm for creating ancestors
*/
__global__
void MetropolisResampling(  curandState_t *global_state, 
                            double *weights,                // [N_particles x 1]
                            const int N_particles, 
                            const int iters, 
                            int* ancestors)                 // [N_particles x 1]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Forse solo thread.x? Se no non parallelizza sui blocchi
    // printf("blockDim = %d; blockIdx = %d; threadIdx = %d\n", blockDim.x, blockIdx.x, threadIdx.x);
    if (tid < N_particles)
    {
        Map<VectorXi> ancestors_vec(ancestors, N_particles);
        ancestors_vec.setZero();
        #if VERBOSE
        if (tid == 0)
        {
            printf("[MR] tid is: %d. N_particles is %d. Iters is %d\n", tid, N_particles, iters);
        }
        #endif
        curandState local_state = global_state[tid];

        int k = tid;
        for (int t = 0; t < iters; t++)
        {
            double u = curand_uniform_double(&local_state);
            double jd = curand_uniform_double(&local_state);
            jd *= (N_particles - 1 + 0.999999);
            int j = (int)trunc(jd);
            if ( u <= weights[j] / weights[k] && !isnan(weights[j] / weights[k]) )
            {
                k = j;
            }
            

        }
        ancestors_vec(tid) = k;
        global_state[tid] = local_state;
    }
}

__global__
void PermutateStatesAndWeights(const cuData &data, double *x_t, double *w_x_t, const int* a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_x)
    {
        #if VERBOSE
        if (tid == 0)
        {
            printf("[PERM_STATES] tid = %d\n", tid);
        }
        #endif
        
        Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
        if (w_x_t_vec.sum() < 1e-200)
        {
            // If weights are all zero, re-initialize them as 1 / N_x and return.
            w_x_t[tid] = 1.0 / data.data.N_x;
            // printf("[PERM_STATES] a contains: %d %d %d %d.\tw_x[tid]: %e; sum(w_x): %e\n", a[0], a[1], a[2], a[3], w_x_t[tid], w_x_t_vec.sum());
            return;
        }
        // printf("[PERM_STATES] a contains: %d %d %d %d.\tw_x[tid]: %e; sum(w_x): %e\n", a[0], a[1], a[2], a[3], w_x_t[tid], w_x_t_vec.sum());

        double *x_t_old, *w_x_t_old;
        cudaMalloc((void**)&x_t_old, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_t_old, sizeof(double) * data.data.N_x);
        memcpy(x_t_old, x_t, sizeof(double) * data.data.N_x);
        memcpy(w_x_t_old, w_x_t, sizeof(double) * data.data.N_x);
        // cudaMemcpyAsync(x_t_old, x_t, sizeof(double) * data.data.N_x, cudaMemcpyDeviceToDevice);
        // cudaMemcpyAsync(w_x_t_old, w_x_t, sizeof(double) * data.data.N_x, cudaMemcpyDeviceToDevice);
        __syncthreads();
        
        x_t[tid] = x_t_old[a[tid]];
        w_x_t[tid] = w_x_t_old[a[tid]];
        // printf("[PERM_STATES] w_x_perm[%d]: %e; w_x_t_old[%d]: %e\n", tid, w_x_t[tid], tid, w_x_t_old[tid]);
        cudaFree(x_t_old);
        cudaFree(w_x_t_old);
    }
}

__global__
void FinalizePFPMMH(const cuData &data,
                    const int T_current,
                    double *x,              // [N_x x (T_current + 1)] All x-particles for each time instant
                    double *w_x,            // Matrix of N_x weights for (T_current + 1) steps
                    double *mlh_hat,        // [1 x 1] Marginal LH referred to this theta vector
                    double *x_hat,          // [(T_current + 1) x 1] Time series for this theta vector
                    double *x_particles,    // [N_x x 1]
                    double *w_x_particles)  // Vector of only the last N_x weights
{
    // Transform the N_x x-particles from t=1:T_current into x_hat. 
    // Compute mlh_hat and save last generated x-particles and weights.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        #if VERBOSE
        printf("[FIN_PMMH] tid: %d\n", tid);
        #endif
        // printf("[FIN_PMMH] w_x is:\n");
        // print_matrix(data.data.N_x, (T_current + 1), w_x, data.data.N_x);
        // Average the states over the weights and return time series, 
        Map<MatrixXd> x_mat(x, data.data.N_x, (T_current + 1));
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (T_current + 1));
        // Map<const::VectorXd> mlh_hat_vec(mlh_hat, data.data.N_theta);
        // printf("[FIN_PMMH] x is\n");
        // print_matrix(data.data.N_x, (T_current + 1), x, data.data.N_x);
        VectorXd w_x_summed = w_x_mat.colwise().sum(); // Along columns: [1 x N_x] vector
        // *mlh_hat = (w_x_mat.colwise().sum() / data.data.N_x).prod();
        *mlh_hat = (w_x_summed / data.data.N_x).prod();
        // Map<VectorXd> x_hat_vec(x_hat, data.data.N);
        Map<VectorXd> x_hat_vec(x_hat, T_current + 1);
        Map<VectorXd> x_particles_vec(x_particles, data.data.N_x);
        Map<VectorXd> w_x_particles_vec(w_x_particles, data.data.N_x);
        
        VectorXd w_x_tmp = w_x_summed.array().inverse();
        MatrixXd w_hat_normalized = w_x_mat * w_x_tmp.asDiagonal();
        // printf("[FIN_PMMH] w_x_normalized is:\n");
        // print_matrix(data.data.N_x, (T_current + 1), w_hat_normalized.data(), data.data.N_x);
        x_hat_vec = ( w_hat_normalized.array() * x_mat.array() ).colwise().sum();
        // x_hat_vec = ( w_x.array() / w_x.colwise().sum().array() * x.array() ).colwise().sum();
        // x_particles_vec = x_mat.rightCols(1);
        // w_x_particles_vec = w_x_mat.rightCols(1);
        x_particles_vec = x_mat.col(T_current);
        w_x_particles_vec = w_x_mat.col(T_current);
        // printf("x_particles:\n");
        // print_matrix(data.data.N_x, 1, x_particles, data.data.N_x);
        cudaFree(x);
        cudaFree(w_x);
    }
}

__global__
void cudaFreePF(double *L, int *a)
{
    // Just cudaFree() the matrices used in PF kernel
    cudaFree(L);
    cudaFree(a);
}

/**
 * Particle Filter Metropolis Hastings (PMMH)
*/
__global__  // CHILD KERNEL FIRST LEVEL
void ParticleFilterPMMH(double *theta,                  // [2 x 1] One theta vector out of N_theta theta vectors
                        int T_current, 
                        const cuData &data, 
                        curandState_t *global_state, 
                        double *mlh_hat,                // [1 x 1] Marginal LH referred to this particular time series
                        double *x_hat,                  // [(T_current + 1) x 1] Time series referred to this theta vector
                        double *x_particles,            // [N_x x 1] Last x-particles used for propagation
                        double *w_x_particles)          // [N_x x 1]
{
    // This kernel runs for each N_theta theta.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        #if VERBOSE
        printf("[PMMH] tid: %d\n", tid);
        #endif
        // (Re)Initialize
        double *x, *w_x;
        // cudaMalloc((void**)&x, sizeof(double) * data.data.N_x * data.data.N);
        // cudaMalloc((void**)&w_x, sizeof(double) * data.data.N_x * data.data.N);
        cudaMalloc((void**)&x, sizeof(double) * data.data.N_x * (T_current + 1));
        cudaMalloc((void**)&w_x, sizeof(double) * data.data.N_x * (T_current + 1));
        // Map<MatrixXd> x_mat(x, data.data.N_x, data.data.N);
        // Map<MatrixXd> w_x_mat(w_x, data.data.N_x, data.data.N);
        Map<MatrixXd> x_mat(x, data.data.N_x, (T_current + 1));     // x_hat up to t = T_current
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (T_current + 1));
        x_mat.setZero();
        w_x_mat.setZero();
        w_x_mat.col(0) = VectorXd::Ones(data.data.N_x) / data.data.N_x;

        // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);

        double *K;
        cudaMalloc((void**)&K, sizeof(double) * data.data.N * data.data.N);
        computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta[0], theta[1], K);
        // Map<MatrixXd> K_mat(K, data.data.N, data.data.N);
        // K_mat.setIdentity();
        // printf("K is:\n");
        // print_matrix(data.data.N, data.data.N, K, data.data.N);


        double *L_tmp;
        cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        // cudaMalloc((void**)&L, sizeof(double) * data.data.N * data.data.N);
        cuCholesky(K, data.data.N, L_tmp);
        cudaFree(K);
        // Map<MatrixXd> L_tmp_mat(L_tmp, data.data.N, data.data.N);
        // MatrixXd L = L_tmp_mat.transpose();
        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);

        // printf("L is:\n");
        // print_matrix(data.data.N, data.data.N, L.data(), data.data.N);
        VectorXd rand_var(data.data.N_x);
        curandState local_state = global_state[tid];
        double f_0 = L(0,0) * curand_normal_double(&local_state);
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            rand_var(ii) = curand_normal_double(&local_state) + f_0;
        }
        x_mat.col(0) = rand_var;
        global_state[tid] = local_state;
        int *a;
        cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        // VectorXi a(data.data.N_x);
        cudaStream_t streamPMMH;
        cudaStreamCreateWithFlags(&streamPMMH, cudaStreamNonBlocking);
        __syncthreads();

        for (int t = 1; t < T_current + 1; t++)
        {
            // printf("[PMMH] Time instant %d of %d. Launching propagate kernel from thread %d...\n", t, T_current, tid);
            
            // PropagateState<<<1, 16, 0, cudaStreamFireAndForget>>>(global_state, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data); // THIS WORKS!
            PropagateState<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data);
            
            // Metropolis resampling (__device__ kernel)

            MetropolisResampling<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, w_x_mat.col(t).data(), data.data.N_x, data.data.B, a);
            
            PermutateStatesAndWeights<<<1, data.data.N_x, 0, streamPMMH>>>(data, x_mat.col(t).data(), w_x_mat.col(t).data(), a);
        }

        // FinalizePFPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(data, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        FinalizePFPMMH<<<1, 1, 0, streamPMMH>>>(data, T_current, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(L.data(), a);
        cudaStreamDestroy(streamPMMH);
        __syncthreads();
    }

}

__global__
void FinalizePF(const cuData &data,
                const int T_next,
                double *x_predicted,            // PF one-step prediction
                double *w_x_predicted,          // PF predicted last-step weights
                double *mlh_hat,                // PF estimated mlh until step t updated to t+1
                double *x_hat_theta,            // Updated trajectory at t+1 for each theta
                double *x_particles,            // N_x particles for next iteration
                double *w_x_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        #if VERBOSE
        printf("[FIN_PF] tid: %d\n", tid);
        #endif
        // Function referred to a single theta pair.
        Map<const::VectorXd> w_x_pred_vec(w_x_predicted, data.data.N_x);
        Map<const::VectorXd> x_pred_vec(x_predicted, data.data.N_x);
        Map<VectorXd> x_hat_theta_mat(x_hat_theta, data.data.N);
        
        *mlh_hat = *mlh_hat / data.data.N_x * w_x_pred_vec.sum();
        double x_t_plus_one = ((w_x_pred_vec.array() * x_pred_vec.array()) / w_x_pred_vec.sum()).sum();
        x_hat_theta_mat(T_next) = x_t_plus_one;
        
        memcpy(x_particles, x_predicted, sizeof(double) * data.data.N_x);
        memcpy(w_x_particles, w_x_predicted, sizeof(double) * data.data.N_x);
        cudaFree(x_predicted);
        cudaFree(w_x_predicted);
    }
}

/**
 * Bootstrap Particle Filter for one-step ahead prediction
*/

__global__
void ParticleFilter(double *theta,                  // [2 x N_theta] Matrix with all thetas
                    const int T_next, 
                    const cuData &data, 
                    curandState_t *global_state, 
                    double *mlh_hat,                // [N_theta x 1] Marginal LH for each theta 
                    double *x_hat,                  // [N x N_theta] Time series for each theta
                    double *x_particles,            // [N_x x N_theta] N_x particles obtained in the last step
                    double *w_x_particles)
{
    // This prediction step has to be done for each theta. States and weights are N x N_theta matrices for this reason!
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_theta)
    {
        #if VERBOSE
        if (tid == 0)
        {
            printf("[PF] tid: %d\n", tid);
        }
        #endif
        // x_particles are the N_x last-step particles for each N_theta
        Map<MatrixXd> x_particles_mat(x_particles, data.data.N_x, data.data.N_theta);
        Map<MatrixXd> w_x_particles_mat(w_x_particles, data.data.N_x, data.data.N_theta);
        Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
        // Map<VectorXd> system_x(data.data.X, data.data.N);

        // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);
        // MatrixXd K = computeKernel(system_x, system_x, theta[0], theta[1]) + 1e-6 * MatrixXd::Identity(data.data.N, data.data.N);
        
        double *K;
        cudaMalloc((void**)&K, sizeof(double) * data.data.N * data.data.N);
        computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_mat(0, tid), theta_mat(1, tid), K);

        double *L_tmp;
        cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        cuCholesky(K, data.data.N, L_tmp);
        cudaFree(K);
        // MatrixXd L(data.data.N, data.data.N);
        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);
        // MatrixXd L = L_mat.transpose();

        int *a;
        cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        // VectorXi a(data.data.N_x);
        cudaStream_t streamPF;
        cudaStreamCreateWithFlags(&streamPF, cudaStreamNonBlocking);
        __syncthreads();

        // Propagate for each particle (__device__ kernel)
        // This is one-step ahead prediction
        double *x_predicted, *w_x_predicted;
        cudaMalloc((void**)&x_predicted, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_predicted, sizeof(double) * data.data.N_x);

        PropagateState<<<1, data.data.N_x, 0, streamPF>>>(global_state, T_next, x_predicted, w_x_predicted, L.data(), data);
        // Metropolis resampling (__device__ kernel)

        MetropolisResampling<<<1, data.data.N_x, 0, streamPF>>>(global_state, w_x_predicted, data.data.N_x, data.data.B, a);

        PermutateStatesAndWeights<<<1, data.data.N_x, 0, streamPF>>>(data, x_predicted, w_x_predicted, a);

        // FinalizePF<<<1, 1, 0, cudaStreamTailLaunch>>>
        //         (data, T_next, x_predicted, w_x_predicted, &mlh_hat[tid], &x_hat[data.data.N * tid], &x_particles[data.data.N_x * tid], &w_x_particles[data.data.N_x * tid]);
        FinalizePF<<<1, 1, 0, streamPF>>>
                (data, T_next, x_predicted, w_x_predicted, &mlh_hat[tid], &x_hat[data.data.N * tid], &x_particles[data.data.N_x * tid], &w_x_particles[data.data.N_x * tid]);

        cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(L.data(), a);
    }

}

/**
 * Initialize needed quantities
*/
__global__
void SMC2_init( curandState *global_state, 
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
    // curand_init(1234, 0, 0, &state[tid]);
    curandState local_state = global_state[tid];
    Map<MatrixXd> system_x(data.data.X, data.data.N, 1);
    Map<MatrixXd> f_mat(f, data.data.N, data.data.N_theta);
    Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
    Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
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
            theta_mat(ii, tid) = 1.0 + curand_normal_double(&local_state);
        }
        w_theta_mat.col(0) = VectorXd::Ones(data.data.N_theta) / data.data.N_theta;

        // Sample N_theta x0 for each theta-particle
        double *K;
        cudaMalloc((void**)&K, sizeof(double) * data.data.N * data.data.N);
        computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta[0], theta[1], K);
        // MatrixXd L = K.llt().matrixL();
        MatrixXd L = MatrixXd::Zero(data.data.N, data.data.N);
        cuCholesky(K, data.data.N, L.data());
        VectorXd rand_var(data.data.N);
        for (int kk = 0; kk < data.data.N; kk++)
        {
            rand_var(kk) = curand_normal_double(&local_state);
        }
        f_mat(0, tid) = L(0, 0) * rand_var(0);
        __syncthreads();
        // Create the particle ensemble at time=0
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            f_particles_mat(ii, tid) = f_mat(0, tid) + curand_normal_double(&local_state);
        }
        __syncthreads();
        // MatrixXd rand_mat(data.data.N_x, data.data.N_theta);
        // for (int ii = 0; ii < data.data.N_x; ii++)
        // {
        //     for (int kk = 0; kk < data.data.N_theta; kk++)
        //     {
        //         rand_mat(ii, kk) = curand_normal_double(&local_state);
        //     }
        // }
        // f_particles_mat = /*f_mat.row(0) +*/ rand_mat;

        // w_f_mat = MatrixXd::Ones(data.data.N_x, data.data.N_theta) / data.data.N_x;
        w_f_mat.setConstant(1.0 / data.data.N_x);

        global_state[tid] = local_state;
        cudaFree(K);
    }
}

__global__
void PermutateThetaAndWeights(  const cuData &data, 
                                double *theta,          // [2 x N_theta] matrix of parameters a t=T_current
                                double *x_hat,          // [N x N_theta] State trajectory t=1:T_current
                                double *mlh,            // [N_theta x 1] Marginal LH at current time
                                double *x_particles,    // [N_x x N_theta] N_x particles for each theta
                                double *w_x_particles,  // [N_x x N_theta] Weights
                                const int *a            // [N_theta x 1] Ancestors
                                )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockDim = %d; blockIdx = %d; threadIdx = %d; tid = %d\n", blockDim.x, blockIdx.x, threadIdx.x, tid);
    if (tid < data.data.N_theta)
    {
        // For each theta, copy the values of input args and then shuffle them using vector a.
        #if VERBOSE
        if (tid == 0)
        {
            printf("[PERM_THETA] Permutate tid = %d\n", tid);
        }
        #endif
        // printf("a_theta contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
        double *mlh_old;
        Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
        if (mlh_vec.sum() < 1e-100)
        {
            mlh[tid] = 1.0 / data.data.N_theta;
            // printf("Re-initializing theta-weights...\n");
        }
        else
        {
            cudaMalloc((void**)&mlh_old, sizeof(double) * data.data.N_theta);
            memcpy(mlh_old, mlh, sizeof(double) * data.data.N_theta);
            memcpy(&mlh[tid], &mlh_old[a[tid]], sizeof(double));
            cudaFree(mlh_old);
        }
        // printf("[PERM_THETA] a contains: %d %d %d %d.\tmlh[tid]: %e; sum(mlh): %e\n", a[0], a[1], a[2], a[3], mlh[tid], mlh_vec.sum());
        double *theta_old, *x_hat_old, *x_particles_old, *w_x_particles_old;
        cudaMalloc((void**)&theta_old, sizeof(double) * 2 * data.data.N_theta);
        cudaMalloc((void**)&x_hat_old, sizeof(double) * data.data.N * data.data.N_theta);
        
        cudaMalloc((void**)&x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
        cudaMalloc((void**)&w_x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
        // double theta_old[2 * data.data.N_theta], x_hat_old[data.data.N * data.data.N_theta], mlh_old[data.data.N_theta], x_particles_old[data.data.N_x * data.data.N_theta], w_x_particles_old[data.data.N_x * data.data.N_theta];
        memcpy(theta_old, theta, sizeof(double) * 2 * data.data.N_theta);
        memcpy(x_hat_old, x_hat, sizeof(double) * data.data.N * data.data.N_theta);
        
        memcpy(x_particles_old, x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
        memcpy(w_x_particles_old, w_x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
        // Copy elements back to input arguments but shuffled using vector a
        memcpy(&theta[2*tid], &theta_old[2*a[tid]], sizeof(double) * 2);
        memcpy(&x_hat[data.data.N * tid], &x_hat_old[data.data.N * a[tid]], sizeof(double) * data.data.N);
        
        memcpy(&x_particles[data.data.N_x*tid], &x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
        memcpy(&w_x_particles[data.data.N_x*tid], &w_x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
        cudaFree(theta_old);
        cudaFree(x_hat_old);
        
        cudaFree(x_particles_old);
        cudaFree(w_x_particles_old);
    }
}

__global__
void NormalizeWeights(  const cuData &data,
                        const double *mlh_hat,      // [N_theta x 1] Marginal LH at current time
                        double *w_theta)            // [N_theta x 1] (Normalized) theta weights at time T_next
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data.data.N_theta)
    {
        #if VERBOSE
        if (tid == 0)
        {
            printf("[NW] tid: %d\n", tid);
        }
        #endif
        Map<const::VectorXd> mlh_hat_vec(mlh_hat, data.data.N_theta);
        double sum_mlh = mlh_hat_vec.sum();
        w_theta[tid] = mlh_hat_vec(tid) / sum_mlh;
    }
}

/**
 * Sequential Monte Carlo² (SMC²) algorithm
*/
void SMC2(const Data &data)
{
    curandState *devStates_theta, *devStates_x;
    // int totalThreads = 256;// data.N_theta * data.N_x;// 256;
    CUDA_CHECK(cudaMalloc((void **)&devStates_theta, data.N_theta * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc((void **)&devStates_x, data.N_x * sizeof(curandState)));
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

    int B = data.B;
    MatrixXd w_theta = MatrixXd::Zero(data.N_theta, data.N);
    MatrixXd theta = MatrixXd::Zero(2, data.N_theta);
    MatrixXd f = MatrixXd::Zero(data.N, data.N_theta);
    Map<VectorXd> system_x(data.X, data.N);
    Map<VectorXd> system_y(data.Y, data.N);

    // Initialize
    setup_curand_theta<<<1, data.N_theta>>>(devStates_theta);
    setup_curand_x<<<1, data.N_x>>>(devStates_x);
    SMC2_init<<<1, data.N_theta>>>(devStates_theta, *dev_data, dev_theta, dev_w_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f);
    cudaDeviceSynchronize();
    // MatrixXd h_f_parts(data.N_x, data.N_theta);
    // CUDA_CHECK(cudaMemcpy(h_f_parts.data(), dev_f_particles, sizeof(double) * data.N_x * data.N_theta, cudaMemcpyDeviceToHost));
    // std::cout << "dev_w_f after initialization:\n" << h_f_parts << std::endl;
    VectorXd x_final(data.N);
    #if PLOT
    plt::figure(1);
    plt::figure(2);
    #endif
    std::cout << "Init done!\n";
    // Cycle through time
    for (int t = 0; t < data.N - 1; t++)
    {
        MatrixXd h_theta(2, data.N_theta), h_f(data.N, data.N_theta);
        // CUDA_CHECK(cudaMemcpy(h_theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "Theta non-permutated:\n" << h_theta << std::endl;
        std::cout << "\x1B[34m============================== Time: " << t << " ==============================\n\x1B[0m";

        MetropolisResampling<<<1, data.N_theta>>>(devStates_theta, &dev_w_theta[t * dev_data->data.N_theta], dev_data->data.N_theta, B, dev_ancestors);
        // MetropolisResampling<<<data.N_theta, 1>>>(devStates_theta, &dev_w_theta[t * dev_data->data.N_theta], dev_data->data.N_theta, B, dev_ancestors);
        // VectorXi h_a(data.N_theta);
        // CUDA_CHECK(cudaMemcpy(h_a.data(), dev_ancestors, sizeof(int) * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "Ancestors:\n" << h_a.transpose() << std::endl;

        // PermutateThetaAndWeights<<<1, data.N_theta>>>(*dev_data, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f, dev_ancestors);
        PermutateThetaAndWeights<<<data.N_theta, 1>>>(*dev_data, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f, dev_ancestors);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // CUDA_CHECK(cudaMemcpy(h_theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "Theta permutated:\n" << h_theta << std::endl;

        // MarginalMetropolisHastings<<<1, data.N_theta>>>(devStates_theta, devStates_x, t, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f, *dev_data);
        MarginalMetropolisHastings<<<data.N_theta, 1>>>(devStates_theta, devStates_x, t, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f, *dev_data);
        // CUDA_CHECK(cudaDeviceSynchronize());
        const int T_next = t + 1;
        // ParticleFilter<<<1, data.N_theta>>>(dev_theta, T_next, *dev_data, devStates_x, dev_mlh, dev_f, dev_f_particles, dev_w_f);
        ParticleFilter<<<data.N_theta, 1>>>(dev_theta, T_next, *dev_data, devStates_x, dev_mlh, dev_f, dev_f_particles, dev_w_f);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // NormalizeWeights<<<1, data.N_theta>>>(*dev_data, dev_mlh, &dev_w_theta[T_next * dev_data->data.N_theta]);
        NormalizeWeights<<<data.N_theta, 1>>>(*dev_data, dev_mlh, &dev_w_theta[T_next * dev_data->data.N_theta]);
        CUDA_CHECK(cudaDeviceSynchronize());
        /*
        CUDA_CHECK(cudaDeviceSynchronize());

        VectorXd h_mlh(data.N_theta);
        CUDA_CHECK(cudaMemcpy(h_mlh.data(), dev_mlh, sizeof(double) * data.N_theta, cudaMemcpyDeviceToHost));
        std::cout << "average mlh at time " << T_next << " is:\n" << h_mlh.mean() << std::endl;
        // CUDA_CHECK(cudaMemcpy(h_f.data(), dev_f, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "f full:\n" << h_f << std::endl;
        CUDA_CHECK(cudaMemcpy(f.data(), dev_f, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w_theta.data(), dev_w_theta, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));

        VectorXd x_hat_all = (w_theta.leftCols(T_next+1).transpose().array() * f.topRows(T_next+1).array()).rowwise().sum();
        std::cout << "x_hat:\n" << x_hat_all.transpose() << std::endl;
        VectorXd theta_hat_all(2);
        theta_hat_all(0) = (w_theta.col(T_next).array() * theta.row(0).transpose().array()).sum();
        theta_hat_all(1) = (w_theta.col(T_next).array() * theta.row(1).transpose().array()).sum();
        std::cout << "Theta is:\n" << theta_hat_all.transpose() << std::endl;*/

        #if PLOT
        // plt::figure(1);
        // plt::clf();
        // plt::plot(system_x.head(T_next), system_y.head(T_next), "k+");
        // plt::plot(system_x.head(T_next), x_hat_all);
        // plt::show(false);

        plt::figure(1);
        // plt::clf();
        plt::plot(system_x.segment(t, 2), system_y.segment(t, 2), "k+");
        plt::plot(system_x.segment(t, 2), x_hat_all.segment(t, 2), "b");
        plt::show(false);

        plt::figure(2);
        VectorXd t_vec(1), theta_1(1), theta_2(1);
        t_vec << t; theta_1 << pow(theta_hat_all(0), 2); theta_2 << pow(theta_hat_all(1), 2);

        plt::plot(t_vec, theta_1, "bo");
        plt::plot(t_vec, theta_2, "ro");
        plt::show(false);
        plt::pause(0.05);

        x_final.head(T_next) = x_hat_all;
        #endif
        

        
    }
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    CUDA_CHECK(cudaMemcpy(f.data(), dev_f, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w_theta.data(), dev_w_theta, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));

    VectorXd x_hat_all = (w_theta.transpose().array() * f.array()).rowwise().sum();
    std::cout << "x_hat:\n" << x_hat_all.transpose() << std::endl;
    // #if PLOT
    plt::figure(3);
    plt::plot(system_x, system_y, "k+");
    // plt::plot(system_x, x_final);
    plt::plot(system_x, x_hat_all);
    plt::show(true);
    // #endif

}