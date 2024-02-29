#include "utilsSMC.hpp"

// __host__ __device__
// MatrixXd computeKernel( Eigen::MatrixXd x1,
//                         Eigen::MatrixXd x2, 
//                         const double amplitude, 
//                         const double length_scale
//                         )
// {
//     double l = length_scale;

//     MatrixXd kernel(x1.rows(), x2.rows()); // Initialize kernel matrix
//     for (int ii = 0; ii < x1.rows(); ii++)
//     {
//         for (int jj = 0; jj < x2.rows(); jj++)
//         {
//             kernel(ii,jj) = amplitude * amplitude * exp( -0.5 / l / l * (x1.row(ii) - x2.row(jj)) * (x1.row(ii) - x2.row(jj)).transpose() );
//         }
//     }
//     return kernel;
// };

/**
 * Compute kernel for unidimensional problems
*/
__host__ __device__
void computeKernel( const double *x1,
                    const int n,
                    const double *x2, 
                    const int m,
                    const double amplitude, 
                    const double l,
                    double *K
                    )
{
    Map<MatrixXd> K_mat(K, n, m);
    for (int ii = 0; ii < n; ii++)
    {
        for (int jj = 0; jj < m; jj++)
        {
            K_mat(ii,jj) = amplitude * amplitude * exp( -0.5 / l / l * (x1[ii] - x2[jj]) * (x1[ii] - x2[jj]) );
        }
    }
    K_mat += 1e-9 * MatrixXd::Identity(n, m);
};

/**
 * Naïve Cholesky decomposition for GPU
*/
__device__
void cuCholesky(const double *A, const int lda, double *L)
{
    Map<MatrixXd> L_mat(L, lda, lda);
    L_mat.setZero();
    Map<const::MatrixXd> A_mat(A, lda, lda);
    for (int i = 0; i < lda; i++) 
    {
        for (int j = 0; j <= i; j++) 
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L_mat(i, k) * L_mat(j, k);

            if (i == j)
                L_mat(i, j) = sqrt(A_mat(i, i) - sum);
            else
                L_mat(i, j) = (1.0 / L_mat(j, j) * (A_mat(i, j) - sum));
            if (isnan(L_mat(i, j)))
            {
                printf("\x1B[33mWarning: L(%d, %d) is nan\n\x1B[0m", i, j);
            }
        }   
    }
}

__global__ 
void setup_curand_theta(curandState *state, const int n)
{
    // Initialize curand with a different state for each thread
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    // curand_init(1234, idx, 0, &state[idx]);
    curand_init(clock64(), idx + (idx * n), 0, &state[idx + (idx * n)]);
}

__global__ 
void setup_curand_x(curandState *state, const int n)
{
    // Initialize curand with a different state for each thread
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    // curand_init(1234, idx, 0, &state[idx]);
    curand_init(clock64(), idx + (idx * n), 0, &state[idx + (idx * n)]);
}

__device__
void print_matrix(const int &m, const int &n, const double *A, const int &lda) 
{
    for (int i = 0; i < m; i++) {
        // printf("|");
        for (int j = 0; j < n; j++) {
            printf("%0.2e\033[%dG", A[j * lda + i], 11 * (j + 1));
            // printf("\033[%dG", 8 * j);
        }
        printf("\n");
    }
}

/*
VectorXd mvn_sampler(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov)
{
    // Multivariate normal sampler using cuRAND

    float *samples;
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * num_samples));
    curandGenerateNormal(gen, samples, num_samples, 0.0, 1.0);

    VectorXf rand_data(num_samples);
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost));

    // Sample from the multivariate normal
    LLT<MatrixXd> llt_dec(cov);
    MatrixXd L = cov.llt().matrixL();
    VectorXd out = mean + L * rand_data.cast<double>();
    return out;
};

VectorXd mvn_sampler_double(curandGenerator_t &gen, int num_samples, VectorXd &mean, MatrixXd &cov)
{
    // Multivariate normal sampler using cuRAND

    double *samples;
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * num_samples));
    curandGenerateNormalDouble(gen, samples, num_samples, 0.0, 1.0);

    VectorXd rand_data(num_samples);
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(double) * num_samples, cudaMemcpyDeviceToHost));

    // Sample from the multivariate normal
    LLT<MatrixXd> llt_dec(cov);
    MatrixXd L = cov.llt().matrixL();
    VectorXd out = mean + L * rand_data;
    return out;
};

VectorXd uniform_sampler(curandGenerator_t &gen, int num_samples)
{
    float *samples2;
    CUDA_CHECK(cudaMalloc((void**)&samples2, sizeof(float) * num_samples));
    CURAND_CHECK(curandGenerateUniform(gen, samples2, num_samples));

    // float *h_samples2 = new float[num_samples];
    VectorXf rand_data(num_samples);
    // CUDA_CHECK(cudaMemcpy(h_samples, samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost));
    cudaError_t error = cudaMemcpy(rand_data.data(), samples2, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);
    std::cout << "Error is " << error << std::endl;
    // Map<VectorXf> rand_data(h_samples2, num_samples);
    cudaFree(samples2);
    return rand_data.cast<double>();
}

VectorXd uniform_sampler_double(curandGenerator_t &gen, int num_samples)
{
    double *samples;
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * num_samples));
    curandGenerateUniformDouble(gen, samples, num_samples);

    double *h_samples;
    CUDA_CHECK(cudaMemcpy(h_samples, samples, sizeof(double) * num_samples, cudaMemcpyDeviceToHost));
    Map<VectorXd> rand_data(h_samples, num_samples);
    return rand_data;
}


VectorXd uni_to_multivariate(const VectorXf &random_samples, const VectorXd &mean, const MatrixXd &cov)
{
    LLT<MatrixXd> llt_dec(cov);
    MatrixXd L = cov.llt().matrixL();
    VectorXd out = mean + L * random_samples.cast<double>();
    return out;
}

VectorXd uni_to_multivariate_double(const VectorXd &random_samples, const VectorXd &mean, const MatrixXd &cov)
{
    LLT<MatrixXd> llt_dec(cov);
    MatrixXd L = cov.llt().matrixL();
    VectorXd out = mean + L * random_samples;
    return out;
}

__global__ 
void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int,  unsigned int *result)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;

  int count = 0;
  while (count < n){
    float myrandf = curand_uniform(my_curandstate+idx);
    myrandf *= (max_rand_int[idx] - min_rand_int[idx]+0.999999);
    myrandf += min_rand_int[idx];
    int myrand = (int)truncf(myrandf);

    assert(myrand <= max_rand_int[idx]);
    assert(myrand >= min_rand_int[idx]);
    result[myrand-min_rand_int[idx]]++;
    count++;}
}
*/