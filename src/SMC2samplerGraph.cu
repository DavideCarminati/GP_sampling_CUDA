#include "SMC2samplerGraph.hpp"


__global__  // CHILD KERNEL FIRST LEVEL
void MetropolisHastingsReject(  curandState_t *state,
                                const cuData &data,
                                const Graph &graph,
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
        size_t delta_t = abs(graph.current - graph.first);
        // Accept or reject new time series using MH
        // skipahead(100, &local_state);
        double u = curand_uniform_double(&local_state);
        if (*mlh_new / *mlh >= u)
        {
            theta[0] = theta_new[0];
            theta[1] = theta_new[1];
            *mlh = *mlh_new;
            if (graph.direction == -1)
            {
                Map<VectorXd> x_theta_new_vec(x_theta_new, (delta_t + 1));
                double *x_theta_flipped;
                cudaMalloc((void**)&x_theta_flipped, sizeof(double) * (delta_t + 1));
                VectorXd x_theta_flipped_vec = x_theta_new_vec.reverse();
                memcpy(x_theta, x_theta_flipped_vec.data(), sizeof(double) * (delta_t + 1));
                cudaFree(x_theta_flipped);
            }
            else
            {
                memcpy(x_theta, x_theta_new, sizeof(double) * (delta_t + 1));
            }
            memcpy(x_particles, x_particles_new, sizeof(double) * data.data.N_x);
            memcpy(w_x_particles, w_x_particles_new, sizeof(double) * data.data.N_x);
            // printf("Accepted.\n");
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
                                const Graph &graph,
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
        // if (tid == 0)
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
        theta_new[0] = 1.0 + curand_normal_double(&local_state) * 1e-3;
        theta_new[1] = 1.0 + curand_normal_double(&local_state) * 1e-3;
        global_state_theta[tid] = local_state;
        
        double *mlh_new;
        double *x_theta_new, *x_particles_new, *w_x_particles_new;
        
        size_t delta_t = abs(graph.current - graph.first);
        cudaMalloc((void**)&x_theta_new, sizeof(double) * (delta_t + 1));
        cudaMalloc((void**)&x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&mlh_new, sizeof(double));
        // ParticleFilterPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(theta_new, T_current, data, global_state_x, mlh_new, x_theta_new, x_particles_new, w_x_particles_new);
        // bool use_global_mem = false;
        size_t dim_buffer = sizeof(double) * data.data.N * data.data.N;
        if (dim_buffer > 48000)
        {
            dim_buffer = sizeof(int);
            // use_global_mem = true;
        }
        ParticleFilterPMMH<<<1, 1, dim_buffer, cudaStreamTailLaunch>>>
                        (theta_new, graph, data, global_state_x, mlh_new, x_theta_new, x_particles_new, w_x_particles_new);
        MetropolisHastingsReject<<<1, 1, 0, cudaStreamTailLaunch>>>
                        (global_state_theta, data, graph, theta_mat.col(tid).data(), theta_new, x_mat.col(tid).data(), x_theta_new, &mlh[tid], mlh_new, 
                        x_particles_mat.col(tid).data(), x_particles_new, w_x_particles_mat.col(tid).data(), w_x_particles_new);
        // __syncthreads();
        // cudaStreamDestroy(streamMMH);

    }
}

__global__  // CHILD KERNEL SECOND LEVEL
void PropagateStatePMMH(curandState_t *global_state,
                        const Graph &graph,
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
    
        double *rand_var;
        cudaMalloc((void**)&rand_var, sizeof(double) * data.data.N);
        Map<VectorXd> rand_var_vec(rand_var, data.data.N);
        // VectorXd rand_var(data.data.N);
        
        for (int ii = 0; ii < data.data.N; ii++)
        {
            rand_var_vec(ii) = curand_normal_double(&local_state);
        }
        int idx = graph.first + graph.direction * T_current;
        // x_t_vec(tid) = L_mat.row(T_current) * rand_var;
        // x_t_vec(tid) = L_mat.row(graph.current) * rand_var_vec;
        x_t_vec(tid) = L_mat.row(idx) * rand_var_vec;
        // w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[graph.current] - x_t_vec(tid), 2) / data.data.Rnoise );
        w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[idx] - x_t_vec(tid), 2) / data.data.Rnoise );
        // printf("[PROP] tid %d. Time: %d (%d). w_x_t is %e; x_t is %f; y is %f\n", tid, graph.current, T_current, w_x_t[tid], x_t[tid], data.data.Y[graph.current]);
        global_state[tid] = local_state;
        cudaFree(rand_var);
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
    extern __shared__ double tmp_buffer[];
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
        double tmp_sum = w_x_t_vec.sum();
        if (tmp_sum < 1e-200)
        {
            // If weights are all zero, re-initialize them as 1 / N_x and return.
            w_x_t[tid] = 1.0 / data.data.N_x;
            // printf("[PERM_STATES] a contains: %d %d %d %d.\tw_x[tid]: %e; sum(w_x): %e\n", a[0], a[1], a[2], a[3], w_x_t[tid], w_x_t_vec.sum());
            return;
        }
        // printf("[PERM_STATES] a contains: %d %d %d %d.\tw_x[tid]: %e; sum(w_x): %e\n", a[0], a[1], a[2], a[3], w_x_t[tid], w_x_t_vec.sum());
        // w_x_t_vec = w_x_t_vec / tmp_sum; // Normalizing x-weights sometimes it gives NaNs
        // memcpy(&tmp_buffer, x_t, sizeof(double) * data.data.N_x);
        tmp_buffer[tid] = x_t[tid];
        __syncthreads();
        x_t[tid] = tmp_buffer[a[tid]];
        // memcpy(&tmp_buffer, w_x_t, sizeof(double) * data.data.N_x);
        tmp_buffer[tid] = w_x_t[tid];
        __syncthreads();
        w_x_t[tid] = tmp_buffer[a[tid]];
        
        // printf("[PERM_STATES] w_x_perm[%d]: %e; w_x_t_old[%d]: %e\n", tid, w_x_t[tid], tid, w_x_t_old[tid]);
    }
}

__global__
void FinalizePFPMMH(const cuData &data,
                    const Graph &graph,
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
        size_t delta_t = abs(graph.current - graph.first);
        Map<MatrixXd> x_mat(x, data.data.N_x, (delta_t + 1));
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (delta_t + 1));
        // Map<const::VectorXd> mlh_hat_vec(mlh_hat, data.data.N_theta);
        // printf("[FIN_PMMH] x is\n");
        // print_matrix(data.data.N_x, (T_current + 1), x, data.data.N_x);
        VectorXd w_x_summed = w_x_mat.colwise().sum(); // Along columns: [1 x N_x] vector
        // *mlh_hat = (w_x_mat.colwise().sum() / data.data.N_x).prod();
        *mlh_hat = (w_x_summed / data.data.N_x).prod();
        // Map<VectorXd> x_hat_vec(x_hat, data.data.N);
        Map<VectorXd> x_hat_vec(x_hat, delta_t + 1);
        Map<VectorXd> x_particles_vec(x_particles, data.data.N_x);
        Map<VectorXd> w_x_particles_vec(w_x_particles, data.data.N_x);
        
        VectorXd w_x_tmp = w_x_summed.array().inverse();
        MatrixXd w_hat_normalized = w_x_mat * w_x_tmp.asDiagonal();
        // printf("[FIN_PMMH] w_x_normalized is:\n");
        // print_matrix(data.data.N_x, (T_current + 1), w_hat_normalized.data(), data.data.N_x);
        x_hat_vec = ( w_hat_normalized.array() * x_mat.array() ).colwise().sum();
        // x_hat_vec = ( w_x.array() / w_x.colwise().sum().array() * x.array() ).colwise().sum();
        x_particles_vec = x_mat.rightCols(1);
        w_x_particles_vec = w_x_mat.rightCols(1);
        // x_particles_vec = x_mat.col(T_current);
        // w_x_particles_vec = w_x_mat.col(T_current);
        // printf("x_particles:\n");
        // print_matrix(data.data.N_x, 1, x_particles, data.data.N_x);
        cudaFree(x);
        cudaFree(w_x);
        // cudaFree(w_x_summed.data());
        // cudaFree(w_x_tmp.data());
        // cudaFree(w_hat_normalized.data());
    }
}

__global__
void cudaFreePF(cudaStream_t stream, double *L, int *a)
{
    // Just cudaFree() the matrices used in PF kernel
    cudaStreamDestroy(stream);
    cudaFree(L);
    cudaFree(a);
}

/**
 * Particle Filter Metropolis Hastings (PMMH)
*/
__global__  // CHILD KERNEL FIRST LEVEL
void ParticleFilterPMMH(double *theta,                  // [2 x 1] One theta vector out of N_theta theta vectors
                        const Graph &graph, 
                        const cuData &data, 
                        curandState_t *global_state, 
                        double *mlh_hat,                // [1 x 1] Marginal LH referred to this particular time series
                        double *x_hat,                  // [(T_current + 1) x 1] Time series referred to this theta vector
                        double *x_particles,            // [N_x x 1] Last x-particles used for propagation
                        double *w_x_particles)          // [N_x x 1]
{
    // This kernel runs for each N_theta theta.
    extern __shared__ double K[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        #if VERBOSE
        printf("[PMMH] tid: %d; %d --> %d; T_curr: %d\n", tid, graph.first, graph.last, graph.current);
        #endif
        // (Re)Initialize
        double *x, *w_x;
        size_t delta_t = abs(graph.current - graph.first);
        cudaMalloc((void**)&x, sizeof(double) * data.data.N_x * (delta_t + 1));
        cudaMalloc((void**)&w_x, sizeof(double) * data.data.N_x * (delta_t + 1));
        
        Map<MatrixXd> x_mat(x, data.data.N_x, (delta_t + 1));     // x_hat up to t = T_current
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (delta_t + 1));
        x_mat.setZero();
        w_x_mat.setZero();
        w_x_mat.col(0) = VectorXd::Ones(data.data.N_x) / data.data.N_x;

        // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);

        size_t dim_buffer = sizeof(double) * data.data.N * data.data.N;
        double *L_tmp;
        cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        if (dim_buffer > 48000)
        {
            // Use global memory to malloc K since it is too big.
            double *K_global;
            cudaMalloc((void**)&K_global, sizeof(double) * data.data.N * data.data.N);
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta[0], theta[1], K_global);
            cuCholesky(K_global, data.data.N, L_tmp);
            cudaFree(K);
        }
        else
        {
            //
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta[0], theta[1], K);
            // Map<MatrixXd> K_mat(K, data.data.N, data.data.N);
            // K_mat.setIdentity();
            // printf("K is:\n");
            // print_matrix(data.data.N, data.data.N, K, data.data.N);

            // cudaMalloc((void**)&L, sizeof(double) * data.data.N * data.data.N);
            cuCholesky(K, data.data.N, L_tmp);
        }
        
        // Map<MatrixXd> L_tmp_mat(L_tmp, data.data.N, data.data.N);
        // MatrixXd L = L_tmp_mat.transpose();
        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);

        // printf("L is:\n");
        // print_matrix(data.data.N, data.data.N, L.data(), data.data.N);

        // BUG BE CAREFUL: EIGEN USES HEAP FOR DYNAMICALLY SIZED MATRICES!
        // VectorXd rand_var(data.data.N_x);
        curandState local_state = global_state[tid];
        double f_0 = L(0,0) * curand_normal_double(&local_state);
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            // rand_var(ii) = curand_normal_double(&local_state) + f_0;
            x_mat(ii, 0) = curand_normal_double(&local_state) + f_0;
        }
        // x_mat.col(0) = rand_var;

        global_state[tid] = local_state;
        int *a;
        cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        // VectorXi a(data.data.N_x);
        cudaStream_t streamPMMH;
        cudaStreamCreateWithFlags(&streamPMMH, cudaStreamNonBlocking);
        // __syncthreads();

        for (int t = 1; t < delta_t + 1; t++)
        {
            // printf("[PMMH] Time instant %d of %d. Launching propagate kernel from thread %d...\n", t, T_current, tid);
            
            // PropagateState<<<1, 16, 0, cudaStreamFireAndForget>>>(global_state, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data); // THIS WORKS!
            PropagateStatePMMH<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, graph, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data);
            
            // Metropolis resampling (__device__ kernel)

            MetropolisResampling<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, w_x_mat.col(t).data(), data.data.N_x, data.data.B, a);
            
            // PermutateStatesAndWeights<<<1, data.data.N_x, 0, streamPMMH>>>(data, x_mat.col(t).data(), w_x_mat.col(t).data(), a);
            PermutateStatesAndWeights<<<1, data.data.N_x, sizeof(double) * data.data.N_x, streamPMMH>>>(data, x_mat.col(t).data(), w_x_mat.col(t).data(), a); // With shared memory
        }

        // FinalizePFPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(data, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        FinalizePFPMMH<<<1, 1, 0, streamPMMH>>>(data, graph, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(streamPMMH, L.data(), a);
        // cudaStreamDestroy(streamPMMH);
        // __syncthreads();
    }

}

__global__  // CHILD KERNEL SECOND LEVEL
void PropagateStatePF(  curandState_t *global_state,
                        const Graph &graph,
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
        curandState local_state = global_state[tid];
        Map<VectorXd> x_t_vec(x_t, data.data.N_x);
        Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
        Map<MatrixXd> L_mat(L, data.data.N, data.data.N);
    
        double *rand_var;
        cudaMalloc((void**)&rand_var, sizeof(double) * data.data.N);
        Map<VectorXd> rand_var_vec(rand_var, data.data.N);
        
        for (int ii = 0; ii < data.data.N; ii++)
        {
            rand_var_vec(ii) = curand_normal_double(&local_state);
        }
        // int idx = graph.first + graph.direction * T_current;
        x_t_vec(tid) = L_mat.row(graph.current) * rand_var_vec;
        // x_t_vec(tid) = L_mat.row(idx) * rand_var_vec;
        w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[graph.current] - x_t_vec(tid), 2) / data.data.Rnoise );
        // w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[idx] - x_t_vec(tid), 2) / data.data.Rnoise );
        
        global_state[tid] = local_state;
        cudaFree(rand_var);
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
void ParticleFilter(curandState_t *global_state, 
                    const cuData &data, 
                    const Graph &graph,
                    double *theta,                  // [2 x N_theta] Matrix with all thetas
                    double *mlh_hat,                // [N_theta x 1] Marginal LH for each theta 
                    double *x_hat,                  // [N x N_theta] Time series for each theta
                    double *x_particles,            // [N_x x N_theta] N_x particles obtained in the last step
                    double *w_x_particles)
{
    // This prediction step has to be done for each theta. States and weights are N x N_theta matrices for this reason!
    extern __shared__ double K[];
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
        
        // double *K;
        // cudaMalloc((void**)&K, sizeof(double) * data.data.N * data.data.N);
        // computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_mat(0, tid), theta_mat(1, tid), K);

        size_t dim_buffer = sizeof(double) * data.data.N * data.data.N;
        double *L_tmp;
        cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        if (dim_buffer > 48000)
        {
            // Use global memory to malloc K since it is too big.
            double *K_global;
            cudaMalloc((void**)&K_global, sizeof(double) * data.data.N * data.data.N);
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_mat(0, tid), theta_mat(1, tid), K_global);
            cuCholesky(K_global, data.data.N, L_tmp);
            cudaFree(K);
        }
        else
        {
            //
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_mat(0, tid), theta_mat(1, tid), K);
            // Map<MatrixXd> K_mat(K, data.data.N, data.data.N);
            // K_mat.setIdentity();
            // printf("K is:\n");
            // print_matrix(data.data.N, data.data.N, K, data.data.N);

            // cudaMalloc((void**)&L, sizeof(double) * data.data.N * data.data.N);
            cuCholesky(K, data.data.N, L_tmp);
        }

        // double *L_tmp;
        // cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        // cuCholesky(K, data.data.N, L_tmp);
        // cudaFree(K);
        
        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);
        

        int *a;
        cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        // VectorXi a(data.data.N_x);
        cudaStream_t streamPF;
        cudaStreamCreateWithFlags(&streamPF, cudaStreamNonBlocking);
        __syncthreads();

        // Propagate for each particle (__device__ kernel)
        // This is one-step ahead prediction
        // double *x_predicted, *w_x_predicted;
        // cudaMalloc((void**)&x_predicted, sizeof(double) * data.data.N_x);
        // cudaMalloc((void**)&w_x_predicted, sizeof(double) * data.data.N_x);
        double *x_predicted = new double[data.data.N_x];
        double *w_x_predicted = new double[data.data.N_x];
        // printf("x_pred points to %p; w_pred to %p\n", x_predicted, w_x_predicted);
        if (!x_predicted || !w_x_predicted)
        {
            printf("[PF] NULL pointers! Out of heap memory!\n");
        }

        PropagateStatePF<<<1, data.data.N_x, 0, streamPF>>>(global_state, graph, x_predicted, w_x_predicted, L.data(), data);
        // Metropolis resampling (__device__ kernel)

        MetropolisResampling<<<1, data.data.N_x, 0, streamPF>>>(global_state, w_x_predicted, data.data.N_x, data.data.B, a);

        // PermutateStatesAndWeights<<<1, data.data.N_x, 0, streamPF>>>(data, x_predicted, w_x_predicted, a);
        PermutateStatesAndWeights<<<1, data.data.N_x, sizeof(double) * data.data.N_x, streamPF>>>(data, x_predicted, w_x_predicted, a); // With shared memory

        // FinalizePF<<<1, 1, 0, cudaStreamTailLaunch>>>
        //         (data, T_next, x_predicted, w_x_predicted, &mlh_hat[tid], &x_hat[data.data.N * tid], &x_particles[data.data.N_x * tid], &w_x_particles[data.data.N_x * tid]);
        FinalizePF<<<1, 1, 0, streamPF>>>
                (data, graph.current, x_predicted, w_x_predicted, &mlh_hat[tid], &x_hat[data.data.N * tid], &x_particles[data.data.N_x * tid], &w_x_particles[data.data.N_x * tid]);

        // cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(streamPF, L.data(), a);
    }

}

__global__
void FinalizePS(const cuData &data,
                const Graph &graph,
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
        printf("[FIN_PS] tid: %d\n", tid);
        #endif

        // Average the states over the weights and return time series, 
        size_t delta_t = abs(graph.current - graph.first);
        Map<MatrixXd> x_mat(x, data.data.N_x, (delta_t + 1));
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (delta_t + 1));
        VectorXd w_x_summed = w_x_mat.colwise().sum(); // Along columns: [1 x N_x] vector
        *mlh_hat = (w_x_summed / data.data.N_x).prod();

        Map<VectorXd> x_hat_vec(x_hat, delta_t + 1);
        Map<VectorXd> x_particles_vec(x_particles, data.data.N_x);
        Map<VectorXd> w_x_particles_vec(w_x_particles, data.data.N_x);
        
        VectorXd w_x_tmp = w_x_summed.array().inverse();
        MatrixXd w_hat_normalized = w_x_mat * w_x_tmp.asDiagonal();

        if (graph.direction == -1)
        {
            VectorXd x_hat_vec_tmp = ( w_hat_normalized.array() * x_mat.array() ).colwise().sum();
            x_hat_vec = x_hat_vec_tmp.reverse();
        }
        else
        {
            x_hat_vec = ( w_hat_normalized.array() * x_mat.array() ).colwise().sum();
        }
        // x_particles_vec = x_mat.rightCols(1);
        // w_x_particles_vec = w_x_mat.rightCols(1);
        x_particles_vec = x_mat.leftCols(1);
        w_x_particles_vec = w_x_mat.leftCols(1);
        cudaFree(x);
        cudaFree(w_x);
    }
}

/**
 * @brief Particle smoother
 * 
 * @param theta 
 * @param graph 
 * @param data 
 * @param global_state 
 * @param mlh_hat 
 * @param x_hat 
 * @param x_particles 
 * @param w_x_particles 
 * @return void
 */

__global__ 
void ParticleSmoother(  double *theta,                  // [2 x N_theta] Whole theta vector
                        double *w_theta,                // [N_theta x N]
                        const Graph &graph, 
                        const cuData &data, 
                        curandState_t *global_state, 
                        double *mlh_smoothed,           // [1 x 1] Marginal LH referred to this particular time series
                        double *x_smoothed,             // [(delta_t + 1) x 1] Time series referred to this theta vector
                        double *x_particles,            // [N_x x N_theta] Last x-particles used for propagation in this branch
                        double *w_x_particles)          // [N_x x N_theta]
{
    // This kernel runs for each N_theta theta.
    extern __shared__ double K[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("\x1B[34m============================== Smoothing %d --> %d ==============================\n\x1B[0m", graph.first, graph.last);
        #if VERBOSE
        printf("[PS] tid: %d; %d --> %d; T_curr: %d\n", tid, graph.first, graph.last, graph.current);
        #endif
        // Compute theta values found in the process
        VectorXd theta_hat(2);
        Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
        Map<MatrixXd> w_theta_mat(w_theta, data.data.N_theta, data.data.N);
        theta_hat(0) = (w_theta_mat.array() * theta_mat.row(0).transpose().array()).sum();
        theta_hat(1) = (w_theta_mat.array() * theta_mat.row(1).transpose().array()).sum();
        // (Re)Initialize
        double *x, *w_x;
        size_t delta_t = abs(graph.last - graph.first);
        cudaMalloc((void**)&x, sizeof(double) * data.data.N_x * (delta_t + 1));
        cudaMalloc((void**)&w_x, sizeof(double) * data.data.N_x * (delta_t + 1));
        
        Map<MatrixXd> x_mat(x, data.data.N_x, (delta_t + 1));     // x_hat up to t = T_current
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, (delta_t + 1));
        x_mat.setZero();
        Map<MatrixXd> x_particles_mat(x_particles, data.data.N_x, data.data.N_theta);
        x_mat.rightCols(1) = (x_particles_mat * w_theta_mat.rightCols(1).asDiagonal()).rowwise().sum() / w_theta_mat.rightCols(1).sum();

        w_x_mat.setZero();
        Map<MatrixXd> w_x_particles_mat(w_x_particles, data.data.N_x, data.data.N_theta);
        w_x_mat.rightCols(1) = (w_x_particles_mat * w_theta_mat.rightCols(1).asDiagonal()).rowwise().sum() / w_theta_mat.rightCols(1).sum();

        // K can be built incrementally at each time instant: K = computeKernel(system_x(0:T_current), system_x(0:T_current), theta[0], theta[1]);

        size_t dim_buffer = sizeof(double) * data.data.N * data.data.N;
        double *L_tmp;
        cudaMalloc((void**)&L_tmp, sizeof(double) * data.data.N * data.data.N);
        if (dim_buffer > 48000)
        {
            // Use global memory to malloc K since it is too big.
            double *K_global;
            cudaMalloc((void**)&K_global, sizeof(double) * data.data.N * data.data.N);
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_hat(0), theta_hat(1), K_global);
            cuCholesky(K_global, data.data.N, L_tmp);
            cudaFree(K);
        }
        else
        {
            computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_hat(0), theta_hat(1), K);
            cuCholesky(K, data.data.N, L_tmp);
        }

        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);

        int *a;
        cudaMalloc((void**)&a, sizeof(int) * data.data.N_x);
        cudaStream_t streamPMMH;
        cudaStreamCreateWithFlags(&streamPMMH, cudaStreamNonBlocking);

        MetropolisResampling<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, w_x_mat.col(delta_t).data(), data.data.N_x, data.data.B, a);
            
        PermutateStatesAndWeights<<<1, data.data.N_x, sizeof(double) * data.data.N_x, streamPMMH>>>(data, x_mat.col(delta_t).data(), w_x_mat.col(delta_t).data(), a);

        for (int t = delta_t - 1; t >= 0; t--)
        {
            PropagateStatePMMH<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, graph, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data);

            MetropolisResampling<<<1, data.data.N_x, 0, streamPMMH>>>(global_state, w_x_mat.col(t).data(), data.data.N_x, data.data.B, a);
            
            PermutateStatesAndWeights<<<1, data.data.N_x, sizeof(double) * data.data.N_x, streamPMMH>>>(data, x_mat.col(t).data(), w_x_mat.col(t).data(), a);
        }

        FinalizePS<<<1, 1, 0, streamPMMH>>>(data, graph, x_mat.data(), w_x_mat.data(), mlh_smoothed, x_smoothed, x_particles, w_x_particles);
        cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(streamPMMH, L.data(), a);
    }

}

/**
 * Initialize needed quantities
*/
__global__
void SMC2_init( curandState *global_state, 
                const cuData &data, 
                const int *initial_nodes_idx,
                const int num_initial_nodes,
                double *theta, 
                double *w_theta, 
                double *f, 
                double *mlh, 
                Particles *particles        // N_x f-particles for each theta-particle and for each branch and their weights
                )            
{
    // Initialize. For each N_theta theta, do the following:
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data.data.N_theta)
    {
        curandState local_state = global_state[tid];
        // Map<MatrixXd> system_x(data.data.X, data.data.N, 1);
        Map<MatrixXd> f_mat(f, data.data.N, data.data.N_theta);
        Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
        Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
        Map<MatrixXd> w_theta_mat(w_theta, data.data.N_theta, data.data.N);
        // Map<MatrixXd> f_particles_mat(f_particles, data.data.N_x, data.data.N_theta);
        // Map<MatrixXd> w_f_mat(w_f, data.data.N_x, data.data.N_theta);
        Map<const::VectorXi> initial_node_idx_vec(initial_nodes_idx, num_initial_nodes);
        mlh_vec.setOnes();
        f_mat.setZero();
    
        // Sample N_theta theta-particles and their weight
        // double *theta = new double[2];
        for (int ii = 0; ii < 2; ii++)
        {
            theta_mat(ii, tid) = 1.0 + curand_normal_double(&local_state) * 1e-3;
        }
        // w_theta_mat.col(0) = VectorXd::Ones(data.data.N_theta) / data.data.N_theta;
        w_theta_mat.setConstant(1.0 / data.data.N_theta);

        // Sample N_theta x0 for each theta-particle
        double *K;
        cudaMalloc((void**)&K, sizeof(double) * data.data.N * data.data.N);
        // printf("[SMC_INIT] tid: %d; K points to %p\n", tid, K);
        computeKernel(data.data.X, data.data.N, data.data.X, data.data.N, theta_mat(0, tid), theta_mat(1, tid), K);
        // MatrixXd L = K.llt().matrixL();
        MatrixXd L = MatrixXd::Zero(data.data.N, data.data.N);
        cuCholesky(K, data.data.N, L.data());
        VectorXd rand_var(data.data.N);
        // for (int kk = 0; kk < data.data.N; kk++)
        // {
        //     rand_var(kk) = curand_normal_double(&local_state);
        // }
        for (int ii = 0; ii < num_initial_nodes; ii++)
        {
            // f_mat(0, tid) = L(0, 0) * rand_var(0);
            for (int kk = 0; kk < data.data.N; kk++)
            {
                rand_var(kk) = curand_normal_double(&local_state);
            }
            f_mat(initial_node_idx_vec(ii), tid) = L.row(initial_node_idx_vec(ii)) * rand_var;
        }
        __syncthreads();
        for (int kk = 0; kk < 2 * num_initial_nodes; kk++)
        {
            // Create the particle ensemble at time=0
            Map<MatrixXd> f_particles_mat(particles[kk].particles, data.data.N_x, data.data.N_theta);
            Map<MatrixXd> w_f_mat(particles[kk].weights, data.data.N_x, data.data.N_theta);
            for (int ii = 0; ii < data.data.N_x; ii++)
            {
                f_particles_mat(ii, tid) = f_mat(0, tid) + curand_normal_double(&local_state);
            }
            w_f_mat.setConstant(1.0 / data.data.N_x);
        }
        // Create the particle ensemble at time=0
        // for (int ii = 0; ii < data.data.N_x; ii++)
        // {
        //     f_particles_mat(ii, tid) = f_mat(0, tid) + curand_normal_double(&local_state);
        // }
        // w_f_mat.setConstant(1.0 / data.data.N_x);

        __syncthreads();

        global_state[tid] = local_state;
        cudaFree(K);
        // cudaFree(rand_var.data());
        // cudaFree(L.data());
    }
}

// Doing so I have only one address per pointer in the device. If I declare the pointer inside a kernel I will have N different mem addresses!
// __device__ double *theta_tmp, *x_hat_tmp, *mlh_tmp, *x_particles_tmp, *w_x_particles_tmp;

__global__
void PermuteThetaAndWeights(const cuData &data, 
                            const int *a,                   // [N_theta x 1] Ancestors
                            double *theta,                  // [2 x N_theta] matrix of parameters a t=T_current
                            double *x_hat,                  // [N x N_theta] State trajectory t=1:T_current
                            double *mlh,                    // [N_theta x 1] Marginal LH at current time
                            double *x_particles,            // [N_x x N_theta] N_x particles for each theta
                            double *w_x_particles,          // [N_x x N_theta] Weights
                            double *theta_tmp,              // [2 x N_theta] Permuted theta matrix
                            double *x_hat_tmp,              // [N x N_theta] Permuted state trajectory
                            double *mlh_tmp,                // [N_theta x 1] Permuted Marginal LH
                            double *x_particles_tmp,        // [N_x x N_theta] Permuted x-particles
                            double *w_x_particles_tmp,      // [N_x x N_theta] Permuted x-weights
                            const bool use_global_mem
                            )
{
    extern __shared__ double tmp_buffer[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockDim = %d; blockIdx = %d; threadIdx = %d; tid = %d\n", blockDim.x, blockIdx.x, threadIdx.x, tid);
    if (tid < data.data.N_theta)
    {
        if (use_global_mem)
        {
            // double *theta_old, *x_hat_old, *mlh_old, *x_particles_old, *w_x_particles_old;
            // double *theta_tmp, *x_hat_tmp, *mlh_tmp, *x_particles_tmp, *w_x_particles_tmp;
            // For each theta, copy the values of input args and then shuffle them using vector a.
            #if VERBOSE
            if (tid == 0)
            {
                printf("[PERM_THETA] Permutate tid = %d\n", tid);
            }
            #endif
            // printf("a_theta contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
            
            Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
            if (mlh_vec.sum() < 1e-200)
            {
                mlh[tid] = 1.0 / data.data.N_theta;
                // printf("Re-initializing theta-weights...\n");
            }
            else
            {
                // cudaMalloc((void**)&mlh_old, sizeof(double) * data.data.N_theta);
                // memcpy(mlh_old, mlh, sizeof(double) * data.data.N_theta);
                // memcpy(&mlh[tid], &mlh_old[a[tid]], sizeof(double));
                // cudaFree(mlh_old);

                // cudaMalloc((void**)&mlh_tmp, sizeof(double) * data.data.N_theta);
                mlh_tmp[tid] = mlh[a[tid]];
                __syncthreads();
                mlh = mlh_tmp;
                // printf("[PERM_THETA] pid: %d; mlh_tmp pointer: %p\n", tid, (void *)mlh_tmp);
            }
            // printf("[PERM_THETA] a contains: %d %d %d %d.\tmlh[tid]: %e; sum(mlh): %e\n", a[0], a[1], a[2], a[3], mlh[tid], mlh_vec.sum());

            // cudaMalloc((void**)&theta_old, sizeof(double) * 2 * data.data.N_theta);
            // memcpy(theta_old, theta, sizeof(double) * 2 * data.data.N_theta);
            // memcpy(&theta[2*tid], &theta_old[2*a[tid]], sizeof(double) * 2);
            // cudaFree(theta_old);

            // cudaMalloc((void**)&theta_tmp, sizeof(double) * 2 * data.data.N_theta);
            memcpy(&theta_tmp[2*tid], &theta[2*a[tid]], sizeof(double) * 2);
            __syncthreads();
            theta = theta_tmp;

            // cudaMalloc((void**)&x_hat_old, sizeof(double) * data.data.N * data.data.N_theta);
            // memcpy(x_hat_old, x_hat, sizeof(double) * data.data.N * data.data.N_theta);
            // memcpy(&x_hat[data.data.N * tid], &x_hat_old[data.data.N * a[tid]], sizeof(double) * data.data.N);
            // cudaFree(x_hat_old);

            // cudaMalloc((void**)&x_hat_tmp, sizeof(double) * data.data.N * data.data.N_theta);
            memcpy(&x_hat_tmp[data.data.N * tid], &x_hat[data.data.N * a[tid]], sizeof(double) * data.data.N);
            __syncthreads();
            x_hat = x_hat_tmp;

            // cudaMalloc((void**)&x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
            // memcpy(x_particles_old, x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
            // memcpy(&x_particles[data.data.N_x*tid], &x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
            // cudaFree(x_particles_old);

            // cudaMalloc((void**)&x_particles_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
            memcpy(&x_particles_tmp[data.data.N_x*tid], &x_particles[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
            __syncthreads();
            x_particles = x_particles_tmp;

            // cudaMalloc((void**)&w_x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
            // memcpy(w_x_particles_old, w_x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
            // memcpy(&w_x_particles[data.data.N_x*tid], &w_x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
            // cudaFree(w_x_particles_old);

            // cudaMalloc((void**)&w_x_particles_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
            memcpy(&w_x_particles_tmp[data.data.N_x*tid], &w_x_particles[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
            __syncthreads();
            w_x_particles = w_x_particles_tmp;

            __syncthreads();
            if (tid == 0)
            {
                cudaFree(theta_tmp);
                cudaFree(x_hat_tmp);
                cudaFree(x_particles_tmp);
                cudaFree(w_x_particles_tmp);
                cudaFree(mlh_tmp);
            }
            return;

        }
        else
        {
            // For each theta, copy the values of input args and then shuffle them using vector a.
            #if VERBOSE
            if (tid == 0)
            {
                printf("[PERM_THETA] Permutate tid = %d\n", tid);
            }
            #endif
            // printf("a_theta contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
            
            Map<VectorXd> mlh_vec(mlh, data.data.N_theta);
            // if (mlh_vec.sum() < 1e-100)
            if (mlh_vec.sum() < 1e-200)
            {
                mlh[tid] = 1.0 / data.data.N_theta;
                // printf("Re-initializing theta-weights...\n");
            }
            else
            {
                // cudaMalloc((void**)&mlh_old, sizeof(double) * data.data.N_theta);
                memcpy(&tmp_buffer, mlh, sizeof(double) * data.data.N_theta);
                memcpy(&mlh[tid], &tmp_buffer[a[tid]], sizeof(double));
            }
            // printf("[PERM_THETA] a contains: %d %d %d %d.\tmlh[tid]: %e; sum(mlh): %e\n", a[0], a[1], a[2], a[3], mlh[tid], mlh_vec.sum());

            memcpy(&tmp_buffer, theta, sizeof(double) * 2 * data.data.N_theta);
            memcpy(&theta[2*tid], &tmp_buffer[2*a[tid]], sizeof(double) * 2);

            memcpy(&tmp_buffer, x_hat, sizeof(double) * data.data.N * data.data.N_theta);
            memcpy(&x_hat[data.data.N * tid], &tmp_buffer[data.data.N * a[tid]], sizeof(double) * data.data.N);
            
            memcpy(&tmp_buffer, x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
            memcpy(&x_particles[data.data.N_x*tid], &tmp_buffer[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);

            memcpy(&tmp_buffer, w_x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
            memcpy(&w_x_particles[data.data.N_x*tid], &tmp_buffer[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
        }

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

// template <typename T, typename... Arguments>
// __global__
// void cudaGarbageCollector(T first_ptr, Arguments... ptrs)
// {
//     cudaFree(first_ptr);
//     cudaGarbageCollector<<<1, 1, 0, cudaStreamFireAndForget>>>(ptrs...);
// }

__global__
void SMC2node_init( curandState_t *global_state_theta,
                    curandState_t *global_state_x,
                    const cuData &data,
                    Graph &initial_path_graph,
                    double *theta,                      // [2 x N_theta]
                    double *w_theta,                    // [N_theta x N]
                    double *mlh,                        // [N_theta x 1]
                    double *f_hat,                      // [N x N_theta]
                    double *f_particles,                // [N_x x N_theta]
                    double *w_f                         // [N_x x N_theta]
                    )
{
    printf("\x1B[34m============================== Start time: %d ==============================\n\x1B[0m", initial_path_graph.current);

    cudaStream_t streamSMCinit;
    cudaStreamCreateWithFlags(&streamSMCinit, cudaStreamNonBlocking);

    int *ancestors;
    cudaMalloc((void**)&ancestors, sizeof(int) * data.data.N_theta);
    MetropolisResampling<<<1, data.data.N_theta, 0, streamSMCinit>>>
        (global_state_theta, &w_theta[initial_path_graph.current * data.data.N_theta], data.data.N_theta, data.data.B, ancestors);
        
    size_t dim_buffer = max(data.data.N * data.data.N_theta, data.data.N_theta * data.data.N_x)  * sizeof(double);
    bool use_global_mem = false;
    if (dim_buffer > 48000)
    {
        printf("[NODE_INIT] Using global memory. Buffer is %u bytes.\n", dim_buffer);
        dim_buffer = sizeof(int);
        use_global_mem = true;
    }
    // Actually, each instance of this kernel has a different f_particles matrix since it is referred to a particular time instant -- Done!

    // Creating temporary matrices for permutation
    double *theta_tmp, *f_hat_tmp, *mlh_tmp, *f_particles_tmp, *w_f_tmp;
    cudaMalloc((void**)&theta_tmp, sizeof(double) * 2 * data.data.N_theta);
    cudaMalloc((void**)&f_hat_tmp, sizeof(double) * data.data.N * data.data.N_theta);
    cudaMalloc((void**)&mlh_tmp, sizeof(double) * data.data.N_theta);
    cudaMalloc((void**)&f_particles_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
    cudaMalloc((void**)&w_f_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
    PermuteThetaAndWeights<<<1, data.data.N_theta, dim_buffer, streamSMCinit>>>
                (data, ancestors, theta, f_hat, mlh, f_particles, w_f,
                theta_tmp, f_hat_tmp, mlh_tmp, f_particles_tmp, w_f_tmp, use_global_mem); // With shared memory

    // cudaGarbageCollector<<<1, 1, 0, streamSMCinit>>>(theta_tmp, f_hat_tmp, mlh_tmp, f_particles_tmp, w_f_tmp);
        
        
    MarginalMetropolisHastings<<<data.data.N_theta, 1, 0, streamSMCinit>>>
        (global_state_theta, global_state_x, initial_path_graph, theta, f_hat, mlh, f_particles, w_f, data);

    cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(streamSMCinit, nullptr, ancestors);
        
}

__global__
void SMC2run(   curandState_t *global_state_theta,
                curandState_t *global_state_x,
                const cuData &data,
                Graph &local_path_graph,
                double *theta,                      // [2 x N_theta]
                double *w_theta,                    // [N_theta x N]
                double *mlh,                        // [N_theta x 1]
                double *f_hat,                      // [N x N_theta]
                double *f_particles,                // [N_x x N_theta]
                double *w_f                         // [N_x x N_theta]
                )
{
    // Update current time and check if it is bigger than the boundaries
    local_path_graph.current = local_path_graph.current + local_path_graph.direction;
    if (local_path_graph.direction == 1)
    {
        if (local_path_graph.current > local_path_graph.last)
        {
            // Reached boundaries of graph
            return;
        }
    }
    if (local_path_graph.direction == -1)
    {
        if (local_path_graph.current < local_path_graph.last)
        {
            return;
        }
    }

    // Otherwise, recursively call this function until boundaries are reached
    printf("\x1B[34m============================== Time: %d ==============================\n\x1B[0m", local_path_graph.current);
    cudaStream_t streamSMCrun;
    cudaStreamCreateWithFlags(&streamSMCrun, cudaStreamNonBlocking);
    size_t dim_buffer = sizeof(double) * data.data.N * data.data.N;
    // ParticleFilter<<<data.data.N_theta, 1, dim_buffer, streamSMCrun>>>
    //         (global_state_x, data, local_path_graph, theta, mlh, f_hat, f_particles, w_f);
    ParticleFilter<<<16, 16, dim_buffer, streamSMCrun>>>
            (global_state_x, data, local_path_graph, theta, mlh, f_hat, f_particles, w_f);
    NormalizeWeights<<<data.data.N_theta, 1, 0, streamSMCrun>>>
            (data, mlh, &w_theta[local_path_graph.current * data.data.N_theta]);
    // printf("After [NW]\n");

    int *ancestors;
    cudaMalloc((void**)&ancestors, sizeof(int) * data.data.N_theta);
    MetropolisResampling<<<1, data.data.N_theta, 0, streamSMCrun>>>
            (global_state_theta, &w_theta[local_path_graph.current * data.data.N_theta], data.data.N_theta, data.data.B, ancestors);
    // MetropolisResampling<<<data.N_theta, 1>>>(devStates_theta, &dev_w_theta[t * dev_data->data.N_theta], dev_data->data.N_theta, B, dev_ancestors);
    // VectorXi h_a(data.N_theta);
    // CUDA_CHECK(cudaMemcpy(h_a.data(), dev_ancestors, sizeof(int) * data.N_theta, cudaMemcpyDeviceToHost));
    // std::cout << "Ancestors:\n" << h_a.transpose() << std::endl;
    // printf("After [MR]\n");

    dim_buffer = max(data.data.N * data.data.N_theta, data.data.N_theta * data.data.N_x) * sizeof(double);
    bool use_global_mem = false;
    if (dim_buffer > 48000)
    {
        printf("[NODE_RUN] Using global memory. Buffer is %u bytes.\n", dim_buffer);
        dim_buffer = sizeof(int);
        use_global_mem = true;
    }
    // Creating temporary matrices for permutation
    double *theta_tmp, *f_hat_tmp, *mlh_tmp, *f_particles_tmp, *w_f_tmp;
    cudaMalloc((void**)&theta_tmp, sizeof(double) * 2 * data.data.N_theta);
    cudaMalloc((void**)&f_hat_tmp, sizeof(double) * data.data.N * data.data.N_theta);
    cudaMalloc((void**)&mlh_tmp, sizeof(double) * data.data.N_theta);
    cudaMalloc((void**)&f_particles_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
    cudaMalloc((void**)&w_f_tmp, sizeof(double) * data.data.N_x * data.data.N_theta);
    PermuteThetaAndWeights<<<1, data.data.N_theta, dim_buffer, streamSMCrun>>>
                (data, ancestors, theta, f_hat, mlh, f_particles, w_f,
                theta_tmp, f_hat_tmp, mlh_tmp, f_particles_tmp, w_f_tmp, use_global_mem); // With shared memory
    // printf("After [PERM_THETA]\n");

    // CUDA_CHECK(cudaMemcpy(h_theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
    // std::cout << "Theta permutated:\n" << h_theta << std::endl;

    MarginalMetropolisHastings<<<data.data.N_theta, 1, 0, streamSMCrun>>>
            (global_state_theta, global_state_x, local_path_graph, theta, f_hat, mlh, f_particles, w_f, data);
    // MarginalMetropolisHastings<<<1, data.data.N_theta, 0, streamSMCrun>>>
    //         (global_state_theta, global_state_x, local_path_graph, theta, f_hat, mlh, f_particles, w_f, data);
    // printf("After [MMH]\n");
    // SMC2run<<<1, 1, 0, streamSMCrun>>>// taillaunch?
    //         (global_state_theta, global_state_x, data, local_path_graph, theta, w_theta, mlh, f_hat, f_particles, w_f);
    // printf("After [SMC2 run]\n");
    cudaFreePF<<<1, 1, 0, cudaStreamTailLaunch>>>(streamSMCrun, nullptr, ancestors);

    SMC2run<<<1, 1, 0, cudaStreamTailLaunch>>>// taillaunch?
            (global_state_theta, global_state_x, data, local_path_graph, theta, w_theta, mlh, f_hat, f_particles, w_f);
    // printf("After [SMC2 run]\n");
}

/**
 * Sequential Monte Carlo² (SMC²) algorithm
*/
void SMC2(const Data &data)
{
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 3000e6));
    curandState *devStates_theta, *devStates_x;
    
    CUDA_CHECK(cudaMalloc((void **)&devStates_theta, data.N_theta * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc((void **)&devStates_x, data.N_x * sizeof(curandState)));
    double *dev_theta, *dev_w_theta, *dev_f, *dev_mlh;//, *dev_f_particles, *dev_w_f;
    int *dev_ancestors;
    CUDA_CHECK(cudaMalloc((void**)&dev_theta, sizeof(double) * 2 * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_w_theta, sizeof(double) * data.N_theta * data.N));
    CUDA_CHECK(cudaMalloc((void**)&dev_f, sizeof(double) * data.N_theta * data.N));
    CUDA_CHECK(cudaMalloc((void**)&dev_mlh, sizeof(double) * data.N_theta));
    // CUDA_CHECK(cudaMalloc((void**)&dev_f_particles, sizeof(double) * data.N_x * data.N_theta));
    // CUDA_CHECK(cudaMalloc((void**)&dev_w_f, sizeof(double) * data.N_x * data.N_theta));
    CUDA_CHECK(cudaMalloc((void**)&dev_ancestors, sizeof(int) * data.N_theta));
    cuData *dev_data = new cuData;
    dev_data->data = data;
    CUDA_CHECK(cudaMallocManaged((void**)&(dev_data->data.X), sizeof(double) * data.N));
    std::copy(data.X, data.X + sizeof(double) * data.N, dev_data->data.X);
    CUDA_CHECK(cudaMallocManaged((void**)&(dev_data->data.Y), sizeof(double) * data.N));
    std::copy(data.Y, data.Y + sizeof(double) * data.N, dev_data->data.Y);

    // int B = data.B;
    MatrixXd w_theta = MatrixXd::Zero(data.N_theta, data.N);
    MatrixXd theta = MatrixXd::Zero(2, data.N_theta);
    MatrixXd f = MatrixXd::Zero(data.N, data.N_theta);
    Map<VectorXd> system_x(data.X, data.N);
    Map<VectorXd> system_y(data.Y, data.N);

    // Initialize

    auto t_begin = std::chrono::system_clock::now();
    size_t initial_nodes = 1;
    size_t num_branches = initial_nodes * 2;   // For now, only 2 directions (from center to left + from center to right)
    VectorXi initial_node_idx(initial_nodes);
    // Compute the position of the initial nodes
    // size_t branch_length = trunc((data.N - initial_nodes) / (2 * initial_nodes));

    size_t size_intervals = trunc(data.N / initial_nodes);
    size_t rem = data.N % (initial_nodes);
    VectorXd branch_length(num_branches);
    branch_length.setZero();
    int idx = 0, idx_branch = 0;
    int count = rem;
    for (int ii = 0; ii < initial_nodes; ii++)
    {

        branch_length(idx_branch++) = floor((size_intervals) / 2.0);
        initial_node_idx(idx++) = floor((size_intervals) / 2.0) + floor((size_intervals)) * ii + (rem - count);

        if (count > 0)
        {
            branch_length(idx_branch) = floor(size_intervals) - branch_length(idx_branch - 1);
            idx_branch++;
            count--;
        }
        else
        {
            branch_length(idx_branch) = floor(size_intervals) - branch_length(idx_branch - 1) - 1;
            idx_branch++;
        }
            

    }
    
    
    std::cout << "Initial node indices:\n" << initial_node_idx.transpose() << " with branch length = " << branch_length.transpose() << std::endl;

    int *dev_init_nodes_idx;
    CUDA_CHECK(cudaMalloc((void**)&dev_init_nodes_idx, sizeof(int) * initial_nodes));
    CUDA_CHECK(cudaMemcpy(dev_init_nodes_idx, initial_node_idx.data(), sizeof(int) * initial_nodes, cudaMemcpyHostToDevice));

    // double **dev_f_particles[initial_nodes * 2];
    Particles *dev_particles = new Particles[2 * initial_nodes];    // Each branch has its own particles and weights!
    CUDA_CHECK(cudaMallocManaged((void**)&dev_particles, initial_nodes * 2 * sizeof(Particles)));
    for (int ii = 0; ii < initial_nodes * 2; ii++)
    {
        CUDA_CHECK(cudaMallocManaged((void**)&(dev_particles[ii].particles), sizeof(double) * data.N_x * data.N_theta));
        CUDA_CHECK(cudaMallocManaged((void**)&(dev_particles[ii].weights), sizeof(double) * data.N_x * data.N_theta));
    }


    setup_curand_theta<<<1, data.N_theta>>>(devStates_theta, 0);
    setup_curand_x<<<1, data.N_x>>>(devStates_x, 0);
    SMC2_init<<<1, data.N_theta>>>(devStates_theta, *dev_data, dev_init_nodes_idx, initial_nodes, dev_theta, dev_w_theta, dev_f, dev_mlh, dev_particles);
    cudaDeviceSynchronize();
    // for (int ii = 0; ii < initial_nodes * 2; ii++)
    // {
    //     Map<MatrixXd> parts(dev_particles[ii].particles, data.N_x, data.N_theta);
    //     Map<MatrixXd> weights(dev_particles[ii].weights, data.N_x, data.N_theta);
    //     std::cout << "Parts:\n" << parts << "\nWeights:\n" << weights << std::endl;
    // }

    std::cout << "Init done!\n";
    
    Graph *dev_initial_graph = new Graph[initial_nodes];
    CUDA_CHECK(cudaMallocManaged((void**)&dev_initial_graph, initial_nodes * sizeof(Graph)));
    
    for (int ii = 0; ii < initial_nodes; ii++)
    {
        dev_initial_graph[ii].first = initial_node_idx(ii);
        dev_initial_graph[ii].current = initial_node_idx(ii);
    }

    cudaStream_t *streamSMC2_nodes = new cudaStream_t[initial_nodes];

    for (int n = 0; n < initial_nodes; n++)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streamSMC2_nodes[n], cudaStreamNonBlocking));      
        // Trying to have different states for each node/branch I'm estimating
        curandState *devStates_theta_init, *devStates_x_init;
    
        CUDA_CHECK(cudaMallocAsync((void **)&devStates_theta_init, data.N_theta * sizeof(curandState), streamSMC2_nodes[n]));
        CUDA_CHECK(cudaMallocAsync((void **)&devStates_x_init, data.N_x * sizeof(curandState), streamSMC2_nodes[n]));
        setup_curand_theta<<<1, data.N_theta, 0, streamSMC2_nodes[n]>>>(devStates_theta_init, n);
        setup_curand_x<<<1, data.N_x, 0, streamSMC2_nodes[n]>>>(devStates_x_init, n);
            
        // SMC2node_init<<<1, 1, 0, streamSMC2_nodes[n]>>>
        //     (devStates_theta, devStates_x, *dev_data, dev_initial_graph[n], dev_theta, dev_w_theta, dev_mlh, dev_f, dev_particles[n*2].particles, dev_particles[n*2].weights);
        SMC2node_init<<<1, 1, 0, streamSMC2_nodes[n]>>>
            (devStates_theta_init, devStates_x_init, *dev_data, dev_initial_graph[n], dev_theta, dev_w_theta, dev_mlh, dev_f, dev_particles[n*2].particles, dev_particles[n*2].weights);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int n = 0; n < initial_nodes; n++)
    {
        dev_particles[n*2 + 1] = dev_particles[n*2];
        // Map<MatrixXd> parts(dev_particles[n].particles, data.N_x, data.N_theta);
        // Map<MatrixXd> weights(dev_particles[n].weights, data.N_x, data.N_theta);
        // std::cout << "Parts:\n" << parts << "\nWeights:\n" << weights << std::endl;
    }
    // CUDA_CHECK(cudaStreamDestroy(streamSMC2_nodes));

    Graph *dev_graph = new Graph[num_branches];
    CUDA_CHECK(cudaMallocManaged((void**)&dev_graph, num_branches * sizeof(Graph)));
    // int node_num = 0;
    for (int ii = 0; ii < num_branches; ii++)
    {
        dev_graph[ii].first     = initial_node_idx((int)trunc(ii / 2));
        dev_graph[ii].direction = ii % 2 ? 1 : -1;
        dev_graph[ii].last      = initial_node_idx((int)trunc(ii / 2)) + dev_graph[ii].direction * branch_length(ii);
        if (dev_graph[ii].direction == -1)
        {
            // dev_graph[ii].first--;  // With this, I remove the node from PFPMMH recomputation.
        }
        dev_graph[ii].current   = dev_graph[ii].first;
        printf("direction: %d; first: %d; last: %d\n", dev_graph[ii].direction, dev_graph[ii].first, dev_graph[ii].last);
    }

    cudaStream_t *streamSMC2 = new cudaStream_t[num_branches];
    for (int n = 0; n < num_branches; n++)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streamSMC2[n], cudaStreamNonBlocking));
    }
    double *dev_f_smoothed;
    CUDA_CHECK(cudaMalloc((void**)&dev_f_smoothed, sizeof(double) * data.N));

    for (int n = 0; n < num_branches; n++)
    {
        // Start kernels
        
        // Trying to have different states for each node/branch I'm estimating
        curandState *devStates_theta_run, *devStates_x_run;
    
        CUDA_CHECK(cudaMallocAsync((void **)&devStates_theta_run, data.N_theta * sizeof(curandState), streamSMC2[n]));
        CUDA_CHECK(cudaMallocAsync((void **)&devStates_x_run, data.N_x * sizeof(curandState), streamSMC2[n]));
        setup_curand_theta<<<1, data.N_theta, 0, streamSMC2[n]>>>(devStates_theta_run, 2*n);
        setup_curand_x<<<1, data.N_x, 0, streamSMC2[n]>>>(devStates_x_run, 2*n);
        
        // SMC2run<<<1, 1, 0, streamSMC2[n]>>>
        //         (devStates_theta, devStates_x, *dev_data, dev_graph[n], dev_theta, dev_w_theta, dev_mlh, dev_f, dev_particles[n].particles, dev_particles[n].weights);
        SMC2run<<<1, 1, 0, streamSMC2[n]>>>
                (devStates_theta_run, devStates_x_run, *dev_data, dev_graph[n], dev_theta, dev_w_theta, dev_mlh, dev_f, dev_particles[n].particles, dev_particles[n].weights);

        // ParticleSmoother<<<1, 1, sizeof(double) * data.N * data.N, streamSMC2[n]>>>
        //     (dev_theta, dev_w_theta, dev_graph[n], *dev_data, devStates_x_run, dev_mlh, &dev_f_smoothed[dev_graph[n].first], dev_particles[n].particles, dev_particles[n].weights);


    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int n = 0; n < num_branches; n++)
    {
        curandState *devStates_x_run;
        CUDA_CHECK(cudaMallocAsync((void **)&devStates_x_run, data.N_x * sizeof(curandState), streamSMC2[n]));
        setup_curand_x<<<1, data.N_x, 0, streamSMC2[n]>>>(devStates_x_run, 3*n);
        ParticleSmoother<<<1, 1, sizeof(double) * data.N * data.N, streamSMC2[n]>>>
            (dev_theta, dev_w_theta, dev_graph[n], *dev_data, devStates_x_run, dev_mlh, &dev_f_smoothed[dev_graph[n].first], dev_particles[n].particles, dev_particles[n].weights);

    }


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
    auto t_end = std::chrono::system_clock::now();
    std::cout << "It took " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << "ms.\n";
    
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    #if FINAL_PLOT
    VectorXd f_smoothed(data.N);
    CUDA_CHECK(cudaMemcpy(f.data(), dev_f, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w_theta.data(), dev_w_theta, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f_smoothed.data(), dev_f_smoothed, sizeof(double) * data.N, cudaMemcpyDeviceToHost));

    VectorXd x_hat_all = (w_theta.transpose().array() * f.array()).rowwise().sum();
    std::cout << "x_hat:\n" << x_hat_all.transpose() << std::endl;

    VectorXd theta_hat_all(2);
    theta_hat_all(0) = (w_theta.array() * theta.row(0).transpose().array()).sum();
    theta_hat_all(1) = (w_theta.array() * theta.row(1).transpose().array()).sum();
    std::cout << "Theta is:\n" << theta_hat_all.transpose() << std::endl;
    
    plt::figure(3);
    plt::plot(system_x, system_y, "k+");
    MatrixXd nodes(initial_nodes, 2);
    nodes << system_x(initial_node_idx), system_y(initial_node_idx);
    plt::plot(nodes.col(0), nodes.col(1), "ro");
    plt::plot(system_x, f_smoothed, "b--");
    // plt::plot(system_x, x_final);
    plt::plot(system_x, x_hat_all);
    plt::show(true);
    #endif

    CUDA_CHECK(cudaDeviceReset());

}