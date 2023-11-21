#include <iostream>
#include <Eigen/Core>
#include <random>
#include <chrono>
#include <omp.h>

#include "matplotlibcpp_eigen.h"

#include <MHsampler.hpp>
#include "PTsampler.hpp"
using namespace Eigen;
namespace plt = matplotlibcpp;

MatrixXd computeKernel( Eigen::MatrixXd x1,
                        Eigen::MatrixXd x2, 
                        const double amplitude, 
                        const double length_scale
                        )
{
    // Checking input size
    if (x1.cols() != x2.cols()) 
    {
        throw std::invalid_argument("x1 and x2 must have the same number of columns");
    }

    double l = length_scale;

    MatrixXd kernel(x1.rows(), x2.rows()); // Initialize kernel matrix
    for (int ii = 0; ii < x1.rows(); ii++)
    {
        for (int jj = 0; jj < x2.rows(); jj++)
        {
            kernel(ii,jj) = amplitude * amplitude * exp( -0.5 / l / l * (x1.row(ii) - x2.row(jj)) * (x1.row(ii) - x2.row(jj)).transpose() );
        }
    }
    return kernel.matrix();
};

int main(int, char**){
    
    // Create function
    std::random_device rd;
    std::mt19937 gen(rd());  
    double sigma2_n = 0.003 * 0.003;
    std::normal_distribution<double> dis(0, sqrt(sigma2_n));
    // std::gamma_distribution<double> dis(4.0, 20.0);
    std::uniform_real_distribution<double> unif_dist(0.5, 20);
    // std::normal_distribution<double> temp_dist(1.0, )

    int N = 50;

    VectorXd x_real = VectorXd::LinSpaced(N, 0, 1);
    VectorXd noise_vec = VectorXd::Zero(N, 1).unaryExpr([&](double dummy){return 1e-1 * dis(gen);});
    VectorXd y_real = sin(2 * M_PI * x_real.array()).matrix() + cos(5 * M_PI * x_real.array()).matrix() + noise_vec;
    // VectorXd y_real = sin(M_PI * x_real.array()).matrix() +  noise_vec;

    MatrixXd K = computeKernel(x_real, x_real, 1.0, sqrt(exp(-3.2189)));

    Distribution prior, likelihood, proposal;

    Data train_set;
    train_set.x_train = x_real;
    train_set.y_train = y_real;

    prior.mean = VectorXd::Zero(N);
    prior.covariance = K + MatrixXd::Identity(N,N) * 1e-9;

    proposal.mean = prior.mean;
    proposal.covariance = K + MatrixXd::Identity(N,N) * 1e-9;

    likelihood.mean = y_real;
    likelihood.covariance = sigma2_n * MatrixXd::Identity(N, N);

    MHoptions options;
    options.max_iterations  = 50000;
    options.burnin          = 10000;
    options.store_after     = 100;

    int T = 8;
    VectorXd temps_vector(T);
    // temps_vector << 0.5, 0.9, 1, 10, 50, 100, 250, 500, 800, 1000;
    temps_vector << 0.5, 0.9, 1, 2, 3, 4, 8, 10;//, 7, 10;
    // temps_vector = pow( 10.0, (3*((ArrayXd::LinSpaced(T, 1, T)) - 1)/(T - 1.0)) );
    // temps_vector << 1.0;
    PToptions PTopts;
    PTopts.max_iterations   = 10000;
    PTopts.burnin           = 5000;
    PTopts.store_after      = 100;
    PTopts.temperature.resize(T);
    PTopts.temperature      = temps_vector;

    int trials = 5;
    plt::figure();
    auto begin = std::chrono::high_resolution_clock::now();
    for(int jj = 0; jj < trials; jj++)
    {
        // MatrixXd samples = MatrixXd::Zero(1, N);// MHsampler(train_set, prior, likelihood, proposal, options);
        MatrixXd samples = MHsampler(train_set, prior, likelihood, proposal, options);
        VectorXd y_pred = samples.colwise().mean();
        plt::plot(x_real, y_pred, {{"label", "MH"}});
    }
    auto end = std::chrono::high_resolution_clock::now();
    // VectorXd y_pred = samples.colwise().mean();

    plt::plot(x_real, y_real, "+r");
    plt::legend();
    plt::grid(true);
    plt::show(false);

    plt::figure();
    for(int jj = 0; jj < trials; jj++)
    {
        // PTopts.temperature = VectorXd::Zero(T, 1).unaryExpr([&](double dummy){return unif_dist(gen);});
        // Now using parallel tempering
        auto begin_PT = std::chrono::high_resolution_clock::now();
        std::vector<MatrixXd> samplesPT = PTsampler(train_set, prior, likelihood, proposal, PTopts);
        auto end_PT = std::chrono::high_resolution_clock::now();

        MatrixXd mean_at_t(T, N);
        int t = 0;
        for (auto samples_at_t : samplesPT)
        {
            mean_at_t.row(t++) = samples_at_t.colwise().mean();
        }
        VectorXd y_pred_PT = mean_at_t.colwise().mean();
        std::cout << "PT took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_PT - begin_PT).count() << "ms" << std::endl;

        plt::plot(x_real, y_pred_PT, {{"label", "PT"}});
    }

    std::cout << "MH took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    

    
    // plt::plot(x_real, y_pred, {{"label", "MH"}});
    // plt::plot(x_real, y_pred_PT, {{"label", "PT"}});
    plt::plot(x_real, y_real, "+r");
    plt::grid(true);
    plt::legend();
    plt::show();
    
}
