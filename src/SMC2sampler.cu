#include "SMC2sampler.hpp"


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
                                double *w_x_particles_new)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("[MHR] Inside MH reject\n");
        curandState local_state = state[tid];
        // Accept or reject new time series using MH
        double u = curand_uniform_double(&local_state);
        if (*mlh_new / *mlh >= u)
        {
            theta[0] = theta_new[0];
            theta[1] = theta_new[1];
            memcpy(x_theta, x_theta_new, data.data.N);
            *mlh = *mlh_new;
            memcpy(x_particles, x_particles_new, data.data.N);
            memcpy(w_x_particles, w_x_particles_new, data.data.N);
            printf("Accepted.\n");
        }
        state[tid] = local_state;
    }
}
/**
 * Particle rejuvination
*/
__global__
void MarginalMetropolisHastings(curandState_t *global_state, 
                                const int T_current,
                                double *theta, 
                                double *x, 
                                double *mlh, 
                                double *x_particles, 
                                double *w_x_particles, 
                                const cuData &data)
{
    // BUG: x is a N x N_theta matrix!
    // BUG: mlh is a N_theta vector!
    // BUG: x_particles is a N_x x N_theta matrix!
    // BUG: w_x_particles is a N_x x N_theta matrix!
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_theta)
    {
        curandState local_state = global_state[tid];
        Map<MatrixXd> theta_mat(theta, 2, data.data.N_theta);
        Map<MatrixXd> x_mat(x, data.data.N, data.data.N_theta);
        Map<MatrixXd> x_particles_mat(x_particles, data.data.N_x, data.data.N_theta);
        Map<MatrixXd> w_x_particles_mat(w_x_particles, data.data.N_x, data.data.N_theta);
    
        // Sample new candidate theta for the new time series
        double *theta_new = new double[2];
        theta_new[0] = 1.0 + curand_normal_double(&local_state);
        theta_new[1] = 1.0 + curand_normal_double(&local_state);
        global_state[tid] = local_state;
        
        double *mlh_new;
        double *x_theta_new, *x_particles_new, *w_x_particles_new;
        cudaMalloc((void**)&x_theta_new, sizeof(double) * data.data.N);
        cudaMalloc((void**)&x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_particles_new, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&mlh_new, sizeof(double));
        ParticleFilterPMMH<<<1, 1, 0, cudaStreamFireAndForget>>>(theta_new, T_current, data, global_state, mlh_new, x_theta_new, x_particles_new, w_x_particles_new);
        MetropolisHastingsReject<<<1, 1, 0, cudaStreamTailLaunch>>>
                        (global_state, data, theta_mat.col(tid).data(), theta_new, x_mat.col(tid).data(), x_theta_new, &mlh[tid], mlh_new, 
                        x_particles_mat.col(tid).data(), x_particles_new, w_x_particles_mat.col(tid).data(), w_x_particles_new);
        __syncthreads();

    }
}

__global__
void PropagateState(curandState_t *global_state, const int T_current, double *x_t, double *w_x_t, double *L, const cuData &data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_x)
    {
        printf("[PROP] tid %d. Time %d.\n", tid, T_current);
        // printf("N is %d\n", data.data.N);
        // printf("T-th row of L is:\n");
        // printf("L: %f\t%f\n", L[0], L[99]);
        // print_matrix(data.data.N, data.data.N, L, data.data.N);
        // printf("x particles at time %d\n", T_current);
        // print_matrix(data.data.N_x, 1, x_t, data.data.N_x);
        curandState local_state = global_state[tid];
        Map<VectorXd> x_t_vec(x_t, data.data.N_x);
        Map<VectorXd> w_x_t_vec(w_x_t, data.data.N_x);
        Map<MatrixXd> L_vec(L, data.data.N, data.data.N); // Check this
    
        VectorXd rand_var(data.data.N);
        for (int ii = 0; ii < data.data.N; ii++)
        {
            rand_var(ii) = curand_normal_double(&local_state);
        }
        // x_t_vec(tid) = L_vec.col(T_current).transpose() * rand_var;
        x_t_vec(tid) = L_vec.row(T_current) * rand_var;
        w_x_t_vec(tid) = exp( -0.5 * log(2 * M_PI * data.data.Rnoise) - 0.5 * pow(data.data.Y[T_current] - x_t_vec(tid), 2) / data.data.Rnoise );
        global_state[tid] = local_state;
    }
}

/**
 * Metropolis Resempling algorithm for creating ancestors
*/
__global__
void MetropolisResampling(curandState_t *global_state, double *weights, const int N_particles/*or N_x if inside the PF*/, const int iters, int* ancestors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Forse solo thread.x? Se no non parallelizza sui blocchi
    if (tid < N_particles)
    {
        Map<VectorXi> ancestors_vec(ancestors, N_particles);
        ancestors_vec.setZero();
        printf("[MR] tid is: %d. N_particles is %d. Iters is %d\n", tid, N_particles, iters);
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
            ancestors_vec(tid) = k;

        }
        global_state[tid] = local_state;
    }
}

__global__
void PermutateStatesAndWeights(const cuData &data, double *x_t, double *w_x_t, const int* a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < data.data.N_x)
    {
        printf("[PERM] Permutate tid = %d\n", tid);
        // x and w_x are N_x x 1 vectors since they are referred to the current time instant

        // printf("a contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
        double *x_t_old, *w_x_t_old;
        cudaMalloc((void**)&x_t_old, sizeof(double) * data.data.N_x);
        cudaMalloc((void**)&w_x_t_old, sizeof(double) * data.data.N_x);
        memcpy(x_t_old, x_t, data.data.N_x);
        memcpy(w_x_t_old, w_x_t, data.data.N_x);
        // Oppure parallelizzando: if (tid < N_x)...
        x_t[tid] = x_t_old[a[tid]];
        w_x_t[tid] = w_x_t_old[a[tid]];
        cudaFree(x_t_old);
        cudaFree(w_x_t_old);
        // printf("here %d\n", ii);
        // for (int ii = 0; ii < data.data.N_x; ii++)
        // {
        //     x_t[ii] = x_t_old[a[ii]];
        //     w_x_t[ii] = w_x_t_old[a[ii]];
        //     // printf("here %d\n", ii);
        // }
    }
}

__global__
void FinalizePFPMMH(const cuData &data,
                    const double *x,
                    const double *w_x,      // Matrix of N_x weights for all N steps
                    double *mlh_hat, 
                    double *x_hat, 
                    double *x_particles, 
                    double *w_x_particles)  // Vector of only the last N_x weights
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("[FIN_PMMH] tid: %d\n", tid);
        // Average the states over the weights and return time series, 
        Map<const::MatrixXd> x_mat(x, data.data.N_x, data.data.N);
        Map<const::MatrixXd> w_x_mat(w_x, data.data.N_x, data.data.N);
        // Map<const::VectorXd> mlh_hat_vec(mlh_hat, data.data.N_theta);
        // printf("w_x is\n");
        // print_matrix(data.data.N_x, data.data.N, w_x, data.data.N_x);
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("[PMMH] tid: %d\n", tid);
        // (Re)Initialize
        double *x, *w_x;
        cudaMalloc((void**)&x, sizeof(double) * data.data.N_x * data.data.N /*(T_current + 1)*/);
        cudaMalloc((void**)&w_x, sizeof(double) * data.data.N_x * data.data.N /*(T_current + 1)*/);
        Map<MatrixXd> x_mat(x, data.data.N_x, data.data.N /*(T_current + 1)*/);
        Map<MatrixXd> w_x_mat(w_x, data.data.N_x, data.data.N /*(T_current + 1)*/);
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
        // Map<MatrixXd> L_tmp_mat(L_tmp, data.data.N, data.data.N);
        // MatrixXd L = L_tmp_mat.transpose();
        Map<MatrixXd> L(L_tmp, data.data.N, data.data.N);

        // printf("L is:\n");
        // print_matrix(data.data.N, data.data.N, L.data(), data.data.N);
        VectorXd rand_var(data.data.N_x);
        curandState local_state = global_state[tid];
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            rand_var(ii) = curand_normal_double(&local_state) + L(0,0) * curand_normal_double(&local_state);
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
            printf("[PMMH] Time instant %d of %d. Launching propagate kernel from thread %d...\n", t, T_current, tid);
            
            // PropagateState<<<1, 16, 0, cudaStreamFireAndForget>>>(global_state, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data); // THIS WORKS!
            PropagateState<<<1, 16, 0, streamPMMH>>>(global_state, t, x_mat.col(t).data(), w_x_mat.col(t).data(), L.data(), data);
            __syncthreads();
            
            // Metropolis resampling (__device__ kernel)

            MetropolisResampling<<<1, 16, 0, streamPMMH>>>(global_state, w_x_mat.col(t).data(), data.data.N_x, data.data.B, a);
            
            PermutateStatesAndWeights<<<1, 16, 0, streamPMMH>>>(data, x_mat.col(t).data(), w_x_mat.col(t).data(), a);
        }

        FinalizePFPMMH<<<1, 1, 0, cudaStreamTailLaunch>>>(data, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        // FinalizePFPMMH<<<1, 1, 0, streamPMMH>>>(data, x_mat.data(), w_x_mat.data(), mlh_hat, x_hat, x_particles, w_x_particles);
        cudaStreamDestroy(streamPMMH);
        __syncthreads();
    }

}

__global__
void FinalizePF(const cuData &data,
                const int T_next,
                const double *x_predicted,      // PF one-step prediction
                const double *w_x_predicted,    // PF predicted last-step weights
                double *mlh_hat,                // PF estimated mlh until step t updated to t+1
                double *x_hat_theta,            // Updated trajectory at t+1 for each theta
                double *x_particles,            // N_x particles for next iteration
                double *w_x_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        // Function referred to a single theta pair.
        Map<const::VectorXd> w_x_pred_vec(w_x_predicted, data.data.N_x);
        Map<const::VectorXd> x_pred_vec(x_predicted, data.data.N_x);
        Map<VectorXd> x_hat_theta_mat(x_hat_theta, data.data.N);
        
        *mlh_hat = *mlh_hat / data.data.N_x * w_x_pred_vec.sum();
        double x_t_plus_one = (w_x_pred_vec.array() * x_pred_vec.array() / w_x_pred_vec.sum()).sum();
        x_hat_theta_mat(T_next) = x_t_plus_one;
        
        memcpy(x_particles, x_predicted, data.data.N_x);
        memcpy(w_x_particles, w_x_predicted, data.data.N_x);
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

        PropagateState<<<1, 16, 0, streamPF>>>(global_state, T_next, x_predicted, w_x_predicted, L.data(), data);
        // Metropolis resampling (__device__ kernel)

        MetropolisResampling<<<1, 16, 0, streamPF>>>(global_state, w_x_predicted, data.data.N_x, data.data.B, a);

        PermutateStatesAndWeights<<<1, 16, 0, streamPF>>>(data, x_predicted, w_x_predicted, a);

        FinalizePF<<<1, 1, 0, cudaStreamTailLaunch>>>
                (data, T_next, x_predicted, w_x_predicted, &mlh_hat[tid], &x_hat[data.data.N * tid], &x_particles[data.data.N_x * tid], &w_x_particles[data.data.N_x * tid]);

        // *mlh_hat = *mlh_hat / data.data.N_x * w_x.sum();
        // x_hat(T_next) = (w_x.array() * x_hat_new.array() / w_x.sum()).sum();
        // x_particles = x_hat_new;
        // w_x_particles = w_x;
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
        // MatrixXd K = computeKernel(system_x, system_x, theta_mat(tid, 0), theta_mat(tid, 1)) + 1e-6 * MatrixXd::Identity(data.data.N, data.data.N);
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
        MatrixXd rand_mat(data.data.N_x, data.data.N_theta);
        for (int ii = 0; ii < data.data.N_x; ii++)
        {
            for (int kk = 0; kk < data.data.N_theta; kk++)
            {
                rand_mat(ii, kk) = curand_normal_double(&local_state);
            }
        }
        f_particles_mat = f_mat.row(0) + rand_mat;
        w_f_mat = MatrixXd::Ones(data.data.N_x, data.data.N_theta) / data.data.N_x;

        global_state[tid] = local_state;
    }
}

__global__
void PermutateThetaAndWeights(  const cuData &data, 
                                double *theta,          // [2 x N_theta] matrix of parameters a t=T_current
                                double *x_hat,          // [N x N_theta] State trajectory t=1:T_current
                                double *mlh,            // [N_theta x 1] Marginal LH at current time
                                double *x_particles,    // [N_x x N_theta] N_x particles for each theta
                                double *w_x_particles,  // [N_x x N_theta] Weights
                                const int* a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ 
    if (tid < data.data.N_theta)
    {
        // For each theta, copy the values of input args and then shuffle them using vector a.
        printf("[PERM_THETA] Permutate tid = %d\n", tid);
        // printf("a_theta contains: %d %d %d %d\n", a[0], a[1], a[2], a[3]);
        double *theta_old, *x_hat_old, *mlh_old, *x_particles_old, *w_x_particles_old;
        cudaMalloc((void**)&theta_old, sizeof(double) * 2 * data.data.N_theta);
        cudaMalloc((void**)&x_hat_old, sizeof(double) * data.data.N * data.data.N_theta);
        cudaMalloc((void**)&mlh_old, sizeof(double) * data.data.N_theta);
        cudaMalloc((void**)&x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
        cudaMalloc((void**)&w_x_particles_old, sizeof(double) * data.data.N_x * data.data.N_theta);
        // double theta_old[2 * data.data.N_theta], x_hat_old[data.data.N * data.data.N_theta], mlh_old[data.data.N_theta], x_particles_old[data.data.N_x * data.data.N_theta], w_x_particles_old[data.data.N_x * data.data.N_theta];
        memcpy(theta_old, theta, sizeof(double) * 2 * data.data.N_theta);
        memcpy(x_hat_old, x_hat, sizeof(double) * data.data.N * data.data.N_theta);
        memcpy(mlh_old, mlh, sizeof(double) * data.data.N_theta);
        memcpy(x_particles_old, x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
        memcpy(w_x_particles_old, w_x_particles, sizeof(double) * data.data.N_x * data.data.N_theta);
        // Copy elements back to input arguments but shuffled using vector a
        memcpy(&theta[2*tid], &theta_old[2*a[tid]], sizeof(double) * 2);
        memcpy(&x_hat[data.data.N * tid], &x_hat_old[data.data.N * a[tid]], sizeof(double) * data.data.N);
        memcpy(&mlh[tid], &mlh_old[a[tid]], sizeof(double));
        memcpy(&x_particles[data.data.N_x*tid], &x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
        memcpy(&w_x_particles[data.data.N_x*tid], &w_x_particles_old[data.data.N_x*a[tid]], sizeof(double) * data.data.N_x);
        cudaFree(theta_old);
        cudaFree(x_hat_old);
        cudaFree(mlh_old);
        cudaFree(x_particles_old);
        cudaFree(w_x_particles_old);
    }
}

// __global__
// void ResampleThetaParticles(curandState_t *global_state, 
//                             const int T_current,
//                             double *theta_weights,          // [N_theta x N] matrix
//                             const int N_theta,
//                             const int iters,
//                             int* ancestors)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid == 0)
//     {
//         // Only the column referred to the current time is used for creating ancestors
//         MetropolisResampling<<<1, 16>>>(global_state, &theta_weights[T_current * N_theta], N_theta, iters, ancestors);
//         // printf("theta_weights:\n");
//         // print_matrix(N_theta, 10, theta_weights, N_theta);

//     }
// }

__global__
void NormalizeWeights(  const cuData &data,
                        const double *mlh_hat,      // [N_theta x 1] Marginal LH at current time
                        double *w_theta)            // [N_theta x 1] (Normalized) theta weights at time T_next
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data.data.N_theta)
    {
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
    curandState *devStates;
    int totalThreads = 256;// data.N_theta * data.N_x;// 256;
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

    int B = data.B;
    MatrixXd w_theta = MatrixXd::Zero(data.N_theta, data.N);
    MatrixXd theta = MatrixXd::Zero(2, data.N_theta);
    MatrixXd f = MatrixXd::Zero(data.N, data.N_theta);
    Map<VectorXd> system_x(data.X, data.N);
    Map<VectorXd> system_y(data.Y, data.N);

    // Initialize
    setup_curand<<<1, totalThreads>>>(devStates);
    SMC2_init<<<1, 1>>>(devStates, *dev_data, dev_theta, dev_w_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f);
    int numThreads = 
    plt::figure(1);
    std::cout << "Init done!\n";
    // Cycle through time
    for (int t = 0; t < data.N - 1; t++)
    {
        // VectorXi ancestors(data.N_theta);
        MetropolisResampling<<<1, 16>>>(devStates, &dev_w_theta[t * dev_data->data.N_theta], dev_data->data.N_theta, B, dev_ancestors);
        // ResampleThetaParticles<<<1, 1>>>(devStates, t, dev_w_theta, dev_data->data.N_theta, B, dev_ancestors);
        PermutateThetaAndWeights<<<1, 16>>>(*dev_data, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_w_f, dev_ancestors);
        // VectorXi a(data.N_theta);
        // CUDA_CHECK(cudaMemcpy(a.data(), dev_ancestors, sizeof(int) * data.N_theta, cudaMemcpyDeviceToHost));
        // std::cout << "ancestors:\n" << a << std::endl;
        MarginalMetropolisHastings<<<1, 16>>>(devStates, t, dev_theta, dev_f, dev_mlh, dev_f_particles, dev_f_particles, *dev_data);
        cudaDeviceSynchronize();
        const int T_next = t + 1;
        ParticleFilter<<<1, 16>>>(dev_theta, T_next, *dev_data, devStates, dev_mlh, dev_f, dev_f_particles, dev_w_f);
        cudaDeviceSynchronize();
        NormalizeWeights<<<1, 16>>>(*dev_data, dev_mlh, &dev_w_theta[T_next * dev_data->data.N_theta]);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemcpy(f.data(), dev_f, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(theta.data(), dev_theta, sizeof(double) * 2 * data.N_theta, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w_theta.data(), dev_w_theta, sizeof(double) * data.N * data.N_theta, cudaMemcpyDeviceToHost));

        VectorXd x_hat_all = (w_theta.leftCols(T_next).transpose().array() * f.topRows(T_next).array()).rowwise().sum();
        VectorXd theta_hat_all(2);
        theta_hat_all(0) = (w_theta.col(T_next).array() * theta.row(0).array()).sum();
        theta_hat_all(1) = (w_theta.col(T_next).array() * theta.row(1).array()).sum();

        plt::clf();
        plt::plot(system_x.head(T_next), system_y.head(T_next), "k+");
        plt::plot(system_x.head(T_next), x_hat_all);
        plt::show(false);
        plt::pause(0.05);

        
    }
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

}