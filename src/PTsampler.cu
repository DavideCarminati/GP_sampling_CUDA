#include "MHsampler.hpp"

// using namespace Eigen;


std::vector<MatrixXd> PTsampler(Data &data, Distribution &prior, Distribution &likelihood, Distribution &proposal, PToptions opts)
{
    //
    int N = data.x_train.rows();        // Num of training points
    int T = opts.temperature.size();    // Num of temperatures
    std::cout << "Num of temps = " << T << std::endl;
    std::cout << "Temps are:\n" << opts.temperature << std::endl;
    curandGenerator_t gen;

    std::random_device rd;
    std::mt19937 h_gen(rd());
    std::uniform_int_distribution<int> unif_dist(0, T - 1);
    std::uniform_real_distribution<double> unif_real_dist(0.0, 1.0);

    // cudaStream_t stream_1;
    // cudaStreamCreate(&stream_1);

    // double *samples;
    float *samples;
    // cudaMalloc((void**)samples, sizeof(float) * data.x_train.size() * opts.max_iterations);
    // CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(double) * N * (opts.max_iterations + opts.burnin)));
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * N * (opts.max_iterations + opts.burnin) * T));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand() * 25ULL /*1234ULL*/);
    curandStatus_t status = curandGenerateNormal(gen, samples, N * (opts.max_iterations + opts.burnin) * T, 0.0, 1.0);
    // curandStatus_t status = curandGenerateNormalDouble(gen, samples, N * (opts.max_iterations + opts.burnin), 0.0, 1.0);
    std::cout << "curand status: " << status << std::endl;

    // VectorXd rand_data(N * (opts.max_iterations + opts.burnin));
    VectorXf rand_data(N * (opts.max_iterations + opts.burnin) * T);
    // CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(double) * N * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * N * (opts.max_iterations + opts.burnin) * T, cudaMemcpyDeviceToHost));

    // Sample from the normal prior dist for each temperature
    float *samples_for_prior;
    CUDA_CHECK(cudaMalloc((void**)&samples_for_prior, sizeof(float) * N * T));
    curandGenerateNormal(gen, samples, N * T, 0.0, 1.0);
    MatrixXf rand_data_for_prior(T, N);
    CUDA_CHECK(cudaMemcpy(rand_data_for_prior.data(), samples, sizeof(float) * N * T, cudaMemcpyDeviceToHost));
    // MatrixXd f = mvn_sampler(gen, N * T, prior.mean, prior.covariance);
    // VectorXd f = mvn_sampler_double(gen, N, prior.mean, prior.covariance);
    // VectorXd f = prior.mean + prior.covariance.llt().matrixL() * rand_data.cast<double>();

    // Sample all the acceptance thresholds
    // VectorXd acceptanceThr = log(ArrayXd::Zero(opts.max_iterations + opts.burnin, 1).unaryExpr([&](double dummy){return unif_real_dist(h_gen);}));
    VectorXd acceptanceThr = log(uniform_sampler(gen, T * (opts.max_iterations + opts.burnin)).array());
    // VectorXd acceptanceThr = log(uniform_sampler_double(gen, opts.max_iterations + opts.burnin).array());

    int accepted_samples = 0, exchanged_samples = 0;
    int idx = 0;
    std::vector<MatrixXd> PTsamples(T, MatrixXd::Zero((int)floor(opts.max_iterations / opts.store_after), N));
    // MatrixXd PTsamples((int)floor(opts.max_iterations / opts.store_after), N);
    double sigma_n = likelihood.covariance(0,0);
    MatrixXd f = MatrixXd::Zero(T, N);
    VectorXd lh_at_each_t(T);

    // // Log likelihood of a Gamma distribution
    // auto LH_gamma = [](VectorXd f)
    // {
    //     double alpha = 4.0; double beta = 20.0;
    //     int N = f.size();
    //     return alpha * log(beta) * N - N * log(tgamma(alpha)) + (alpha - 1) * f.array().log().sum() - beta * f.array().sum();
    // };

    std::cout << "Iteration:\n";
    for (int i = 0; i < opts.max_iterations + opts.burnin; i++)
    {
        // For each temperature
        #pragma omp parallel for shared(f, lh_at_each_t)
        for (int t = 0; t < T; t++)
        {
            if (i == 0)
            {
                f.row(t) = uni_to_multivariate(rand_data_for_prior.row(t), prior.mean, prior.covariance);
            }
            // VectorXd fnew = mvn_sampler(gen, N, proposal.mean, proposal.covariance);
            VectorXd fnew = uni_to_multivariate(rand_data.segment(i*N + t*N, N), proposal.mean, proposal.covariance);
            // VectorXd fnew = uni_to_multivariate_double(rand_data.segment(i*N, N), proposal.mean, proposal.covariance);
            
            // Evaluate likelihoods
            double lh_new = -0.5 * N * log( 2 * M_PI * sigma_n) - 
                        0.5 / sigma_n * pow((data.y_train - fnew).array(), 2).sum();
            // double lh_new = LH_gamma(fnew);

            double lh_old = -0.5 * N * log( 2 * M_PI * sigma_n) - 
                        0.5 / sigma_n * pow((data.y_train - f.row(t).transpose()).array(), 2).sum();

            double acceptanceProb = lh_new - lh_old;
            if (min(acceptanceProb, 0.0 ) > acceptanceThr(t + i*t))
            {
                f.row(t) = fnew.transpose();
                accepted_samples++;
                // printf("+");
                lh_old = lh_new;
            }
            lh_at_each_t(t) = lh_old;
        }

        // std::cout << "LH at each temp:\n" << lh_at_each_t << std::endl;
        // Exchange samples between temperatures
        if (T > 1)
        {
            int p = unif_dist(h_gen);
            int q = unif_dist(h_gen);
            while (p == q)
            {
                q = unif_dist(h_gen);
            }
            // std::cout << "p and q: " << p << "; " << q << std::endl;
            // double r1 = pow(lh_at_each_t(q) / lh_at_each_t(p), 1 / opts.temperature(p));
            double r1 = 1 / opts.temperature(p) * (lh_at_each_t(q) - lh_at_each_t(p));
            // double r2 = pow(lh_at_each_t(p) / lh_at_each_t(q), 1 / opts.temperature(q));
            double r2 = 1 / opts.temperature(q) * (lh_at_each_t(p) - lh_at_each_t(q));
            // ArrayXd r1 = pow(f.row(q).array() / f.row(p).array(), 1 / opts.temperature(p));
            // ArrayXd r2 = pow(f.row(p).array() / f.row(q).array(), 1 / opts.temperature(q));
            // double alpha_ladder = min(1.0, (r1*r2).maxCoeff());
            // double alpha_ladder = min(1.0, (r1*r2));
            double alpha_ladder = min(0.0, (r1 + r2));
            if (alpha_ladder < log(unif_real_dist(h_gen)))
            {
                VectorXd tmp = f.row(q).transpose();
                f.row(q) = f.row(p);
                f.row(p) = tmp.transpose();
                exchanged_samples++;
            }
            
        }

        if (i > opts.burnin && i % opts.store_after == 0)
        {
            for (int t = 0; t < T; t++)
            {
                PTsamples[t].row(idx) = f.row(t);
            }
            idx++;
        }
        // printf("\e[1K\e[1G%d", i);
        // std::cout << "Iteration " << i;
        
    }
    std::cout << "\nAcceptance rate: " << accepted_samples / (float)(opts.max_iterations + opts.burnin) * 100 << "%." << std::endl;
    std::cout << "Exchange rate: " << exchanged_samples / (float)(opts.max_iterations + opts.burnin) * 100 << "%." << std::endl;
    // std::cout << "Sample size " << PTsamples.rows() << "x" << PTsamples.cols() << std::endl;
    return PTsamples;

}