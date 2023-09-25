#include <iostream>
#include <Eigen/Core>
#include <random>
#include <chrono>

#include "matplotlibcpp_eigen.h"

#include <MHsampler.hpp>
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
    double sigma2_n = 0.3 * 0.3;
    std::normal_distribution<double> dis(0, sqrt(sigma2_n));

    int N = 100;

    VectorXd x_real = VectorXd::LinSpaced(N, 0, 1);
    VectorXd noise_vec = VectorXd::Zero(N, 1).unaryExpr([&](double dummy){return dis(gen);});
    VectorXd y_real = sin(2 * M_PI * x_real.array()).matrix() + cos(5 * M_PI * x_real.array()).matrix() + noise_vec;

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

    int trials = 10;
    plt::figure();
    auto begin = std::chrono::high_resolution_clock::now();
    // for(int jj = 0; jj < trials; jj++)
    // {
        MatrixXd samples = MHsampler(train_set, prior, likelihood, proposal, options);
        // VectorXd y_pred = samples.colwise().mean();
        // plt::plot(x_real, y_pred);
    // }
    auto end = std::chrono::high_resolution_clock::now();
    VectorXd y_pred = samples.colwise().mean();

    std::cout << "MH took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    
    plt::plot(x_real, y_pred);
    plt::plot(x_real, y_real, "or");
    plt::show();
    
}
