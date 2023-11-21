#include "utils.hpp"

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
    float *samples;
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * num_samples));
    curandGenerateUniform(gen, samples, num_samples);

    VectorXf rand_data(num_samples);
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpyAsync(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost, stream));
    return rand_data.cast<double>();
}

VectorXd uniform_sampler_double(curandGenerator_t &gen, int num_samples)
{
    double *samples;
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * num_samples));
    curandGenerateUniformDouble(gen, samples, num_samples);

    VectorXd rand_data(num_samples);
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(double) * num_samples, cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpyAsync(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost, stream));
    return rand_data;
}

// VectorXi uniform_integer_sampler(curandGenerator_t &gen, int num_samples)
// {
//     float *samples;
//     CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * num_samples));
//     curandGenerateUniform(gen, samples, num_samples);

//     VectorXf rand_data(num_samples);
//     CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost));
//     // CUDA_CHECK(cudaMemcpyAsync(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost, stream));
//     return rand_data.cast<double>();
// }

// VectorXi uniform_integer_sampler_double(curandGenerator_t &gen, int num_samples)
// {
//     double *samples;
//     CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * num_samples));
//     curandGenerateUniformDouble(gen, samples, num_samples);

//     VectorXd rand_data(num_samples);
//     CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(double) * num_samples, cudaMemcpyDeviceToHost));
//     // CUDA_CHECK(cudaMemcpyAsync(rand_data.data(), samples, sizeof(float) * num_samples, cudaMemcpyDeviceToHost, stream));
//     return rand_data;
// }


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
void setup_kernel(curandState *state)
{

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
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