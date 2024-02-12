#include "MHsampler.hpp"


MatrixXd MHsampler(Data &data, Distribution &prior, Distribution &likelihood, Distribution &proposal, MHoptions opts)
{
    //
    int N = data.x_train.rows(); // # of training points
    curandGenerator_t gen;

    // cudaStream_t stream_1;
    // cudaStreamCreate(&stream_1);

    // float *samples;
    // // cudaMalloc((void**)samples, sizeof(float) * data.x_train.size() * opts.max_iterations);
    // CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * N * (opts.max_iterations + opts.burnin)));
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // // curandGenerateNormal(gen, samples, N, 0.0, 1.0);
    // curandStatus_t status = curandGenerateNormal(gen, samples, N * (opts.max_iterations + opts.burnin), 0.0, 1.0);
    // std::cout << "curand status: " << status << std::endl;

    // VectorXf rand_data(N * (opts.max_iterations + opts.burnin));
    // CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * N * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpyAsync(rand_data.data(), samples, sizeof(float) * N * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost, stream_1));

    // double *samples;
    float *samples;
    // cudaMalloc((void**)samples, sizeof(float) * data.x_train.size() * opts.max_iterations);
    // CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * N * (opts.max_iterations + opts.burnin)));
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * N * (opts.max_iterations + opts.burnin)));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand() * 25ULL /*1234ULL*/);
    curandStatus_t status = curandGenerateNormal(gen, samples, N * (opts.max_iterations + opts.burnin), 0.0, 1.0);
    // curandStatus_t status = curandGenerateNormalDouble(gen, samples, N * (opts.max_iterations + opts.burnin), 0.0, 1.0);
    std::cout << "curand status: " << status << std::endl;

    // VectorXd rand_data(N * (opts.max_iterations + opts.burnin));
    VectorXf rand_data(N * (opts.max_iterations + opts.burnin));
    // CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(double) * N * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * N * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost));

    // Sample from the multivariate prior
    VectorXd f = mvn_sampler(gen, N, prior.mean, prior.covariance);
    // VectorXd f = mvn_sampler_double(gen, N, prior.mean, prior.covariance);
    // VectorXd f = prior.mean + prior.covariance.llt().matrixL() * rand_data.cast<double>();

    // Sample all the acceptance thresholds
    VectorXd acceptanceThr = log(uniform_sampler(gen, opts.max_iterations + opts.burnin).array());
    // VectorXd acceptanceThr = log(uniform_sampler_double(gen, opts.max_iterations + opts.burnin).array());

    int accepted_samples = 0;
    int idx = 0;
    MatrixXd MHsamples((int)floor(opts.max_iterations / opts.store_after), N);
    double sigma_n = likelihood.covariance(0,0);

    std::cout << "Iteration:\n";
    for (int i = 0; i < opts.max_iterations + opts.burnin; i++)
    {
        // VectorXd fnew = mvn_sampler(gen, N, proposal.mean, proposal.covariance);
        VectorXd fnew = uni_to_multivariate(rand_data.segment(i*N, N), proposal.mean, proposal.covariance);
        // VectorXd fnew = uni_to_multivariate_double(rand_data.segment(i*N, N), proposal.mean, proposal.covariance);
        
        // Evaluate likelihoods
        double lh_new = -0.5 * log( pow(2 * M_PI, N) * sigma_n) - 
                    0.5 / sigma_n * pow((data.y_train - fnew).array(), 2).sum(); // (data.y_train - fnew) * (data.y_train - fnew).transpose();

        double lh_old = -0.5 * log( pow(2 * M_PI, N) * sigma_n) - 
                    0.5 / sigma_n * pow((data.y_train - f).array(), 2).sum();

        double acceptanceProb = lh_new - lh_old;
        if (std::min(acceptanceProb, 0.0 ) > acceptanceThr(i))
        {
            f = fnew;
            accepted_samples++;
        }

        if (i > opts.burnin && i % opts.store_after == 0)
        {
            MHsamples.row(idx++) = f;
        }
        // printf("\e[1K\e[1G%d", i);
        // std::cout << "Iteration " << i;
        
    }
    std::cout << "\nAcceptance rate: " << accepted_samples / (float)opts.max_iterations * 100 << "%." << std::endl;
    std::cout << "Sample size " << MHsamples.rows() << "x" << MHsamples.cols() << std::endl;
    return MHsamples;

}