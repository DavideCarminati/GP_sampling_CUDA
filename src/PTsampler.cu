#include "PTsampler.hpp"


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

    // double *samples;
    float *samples;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 26ULL /*1234ULL*/);
    CUDA_CHECK(cudaMalloc((void**)&samples, sizeof(float) * N * (opts.max_iterations + opts.burnin) * T));
    curandStatus_t status = curandGenerateNormal(gen, samples, N * (opts.max_iterations + opts.burnin) * T, 0.0, 1.0);
    std::cout << "curand status: " << status << std::endl;

    VectorXf rand_data(N * (opts.max_iterations + opts.burnin) * T);
    CUDA_CHECK(cudaMemcpy(rand_data.data(), samples, sizeof(float) * N * (opts.max_iterations + opts.burnin) * T, cudaMemcpyDeviceToHost));
    // Map<VectorXf> rand_data(h_samples, N * (opts.max_iterations + opts.burnin) * T);
    CUDA_CHECK(cudaFree(samples));

    // Sample from the normal prior dist for each temperature
    float *samples_for_prior;
    MatrixXf rand_data_for_prior(T, N);
    CUDA_CHECK(cudaMalloc((void**)&samples_for_prior, sizeof(float) * N * T));
    CURAND_CHECK(curandGenerateNormal(gen, samples_for_prior, N * T, 0.0, 1.0));
    CUDA_CHECK(cudaMemcpy(rand_data_for_prior.data(), samples_for_prior, sizeof(float) * N * T, cudaMemcpyDeviceToHost));
    // Map<MatrixXf> rand_data_for_prior(h_samples_for_prior, T, N);
    std::cout << rand_data_for_prior << std::endl;
    CUDA_CHECK(cudaFree(samples_for_prior));

    // Sample all the acceptance thresholds
    VectorXd acceptanceThr = log(uniform_sampler(gen, T * (opts.max_iterations + opts.burnin)).array());

    // Create samples for hyperparameter prior
    float *samples_for_hyp_prior;
    MatrixXf rand_data_for_hyp_prior(T * (opts.max_iterations + opts.burnin), 2);
    CUDA_CHECK(cudaMalloc((void**)&samples_for_hyp_prior, sizeof(float) * 2 * T * (opts.max_iterations + opts.burnin)));
    CURAND_CHECK(curandGenerateNormal(gen, samples_for_hyp_prior, 2 * T * (opts.max_iterations + opts.burnin), 0.0, 1.0));
    CUDA_CHECK(cudaMemcpy(rand_data_for_hyp_prior.data(), samples_for_hyp_prior, sizeof(float) * 2 * T * (opts.max_iterations + opts.burnin), cudaMemcpyDeviceToHost));
    // Map<MatrixXf> rand_data_for_hyp_prior(h_samples_for_hyp_prior, T, 2);
    std::cout << rand_data_for_hyp_prior.bottomRows(10) << std::endl;
    // CUDA_CHECK(cudaFree(samples_for_hyp_prior));

    // Sample all the acceptance thresholds for hyperparameters
    VectorXd acceptanceThr_hyp = log(uniform_sampler(gen, T * (opts.max_iterations + opts.burnin)).array());

    // Allocate matrices
    int accepted_samples = 0, exchanged_samples = 0;
    int idx = 0;
    std::vector<MatrixXd> PTsamples(T, MatrixXd::Zero((int)floor(opts.max_iterations / opts.store_after), N));
    // MatrixXd PTsamples((int)floor(opts.max_iterations / opts.store_after), N);
    double sigma2_n = likelihood.covariance(0,0);
    MatrixXd f = MatrixXd::Zero(T, N);
    VectorXd lh_at_each_t(T);

    // Log Likelihood for hyperparameter training
    auto LH_hyp = [&data](MatrixXd cov)
    {
        double N = data.x_train.rows();
        return -0.5 * log(pow(2 * M_PI, N) * cov.determinant()) - 0.5 * data.x_train.transpose() * cov.inverse() * data.x_train;
    };

    // Initialize old likelihood for hyperparameters
    VectorXd LH_old_hyp(T), lh_old(T);
    MatrixXd theta(T, 2);
    std::vector<MatrixXd> K_old_all_t(T, MatrixXd::Zero(N, N));
    Distribution proposal_at_each_t[T];
    MatrixXd K_old = computeKernel(data.x_train, data.x_train, exp(0.0), exp(0.0));
    for (int t = 0; t < T; t++)
    {
        K_old_all_t[t] = K_old;
        LH_old_hyp(t) = LH_hyp(K_old);
        proposal_at_each_t[t].covariance = K_old;
        // Initialize old likelihood for posterior
        f.row(t) = uni_to_multivariate(rand_data_for_prior.row(t), prior.mean, prior.covariance);
        lh_old(t) = -0.5 * log( pow(2 * M_PI, N) * sigma2_n) - 
                        0.5 / sigma2_n * pow((data.y_train - f.row(t).transpose()).array(), 2).sum();
    }

    

    std::cout << "Progress:\n";
    for (int i = 0; i < opts.max_iterations + opts.burnin; i++)
    {
        // if ( i % (int)(opts.max_iterations / 100) == 0 || i == 0 )
        // {
        //     printf("\033[2K\r");
        //     int progress = (i / (opts.max_iterations + opts.burnin) * 10);
        //     std::cout << "[" + std::string(progress, '=') + \
        //             std::string(10 - progress, ' ') + "]";
        // }
        // For each temperature
        #pragma omp parallel for shared(f, lh_at_each_t)
        for (int t = 0; t < T; t++)
        {
            /**
             * SAMPLE THE HYPERPARAMETERS
            */
            // Hyperparameters are all sampled outside this loop. We just need to update the kernel
            VectorXd theta_new = exp(rand_data_for_hyp_prior.row(t + i*T).array().cast<double>());
            MatrixXd K_new = computeKernel(data.x_train, data.x_train, theta_new(0), theta_new(1));
            // Compute new log(p(f|theta_new))
            double LH_new_hyp = LH_hyp(K_new + MatrixXd::Identity(N, N) * 1e-6);
            // Compute acceptance probability and accept/reject
            if ( std::min(LH_new_hyp - LH_old_hyp(t), 0.0) > acceptanceThr_hyp(t + i*T) )
            {
                proposal_at_each_t[t].covariance = K_new + MatrixXd::Identity(N,N) * 1e-9;
                LH_old_hyp(t) = LH_new_hyp;
                theta.row(t) = theta_new.transpose();
            }

            /**
             * SAMPLE THE POSTERIOR
            */
            VectorXd fnew = uni_to_multivariate(rand_data.segment(i*N + t*N, N), proposal.mean, proposal_at_each_t[t].covariance);
            // VectorXd fnew = uni_to_multivariate_double(rand_data.segment(i*N, N), proposal.mean, proposal.covariance);
            
            // Evaluate likelihoods
            double lh_new = -0.5 * log( pow(2 * M_PI, N) * sigma2_n) - 
                        0.5 / sigma2_n * pow((data.y_train - fnew).array(), 2).sum();
            // double lh_new = LH_gamma(fnew);

            // double lh_old = -0.5 * N * log( 2 * M_PI * sigma2_n) - 
            //             0.5 / sigma2_n * pow((data.y_train - f.row(t).transpose()).array(), 2).sum();

            double acceptanceProb = lh_new - lh_old(t);
            if (std::min(acceptanceProb, 0.0 ) > acceptanceThr(t + i*T))
            {
                f.row(t) = fnew.transpose();
                accepted_samples++;
                // printf("+");
                lh_old(t) = lh_new;
            }
            lh_at_each_t(t) = lh_old(t);
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
            double alpha_ladder = std::min(0.0, (r1 + r2));
            if (alpha_ladder < log(unif_real_dist(h_gen)))
            {
                VectorXd tmp = f.row(q).transpose();
                f.row(q) = f.row(p);
                f.row(p) = tmp.transpose();
                // Exchange hyperparameters as well
                tmp.resize(2);
                tmp = theta.row(q).transpose();
                theta.row(q) = theta.row(p);
                theta.row(p) = tmp.transpose();
                // And kernels
                MatrixXd tmp_mat = proposal_at_each_t[q].covariance;
                proposal_at_each_t[q].covariance = proposal_at_each_t[p].covariance;
                proposal_at_each_t[p].covariance = tmp_mat;
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
        printf("\e[1K\e[1G%d", i);
        // std::cout << "Iteration " << i;
        
    }
    std::cout << "\nAcceptance rate: " << accepted_samples / (float)(opts.max_iterations + opts.burnin) * 100 << "%." << std::endl;
    std::cout << "Exchange rate: " << exchanged_samples / (float)(opts.max_iterations + opts.burnin) * 100 << "%." << std::endl;
    // std::cout << "Sample size " << PTsamples.rows() << "x" << PTsamples.cols() << std::endl;
    return PTsamples;

}