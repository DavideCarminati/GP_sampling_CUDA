#include <iostream>
#include <Eigen/Core>
#include <random>
#include <chrono>
#include <omp.h>

#include "matplotlibcpp_eigen.h"

#include "SMC2sampler.hpp"
#include "utils.hpp"
using namespace Eigen;
namespace plt = matplotlibcpp;

int main(int, char**){
    
    // Create function
    std::random_device rd;
    std::mt19937 gen(rd());  
    double sigma2_n = 0.01 * 0.01;
    std::normal_distribution<double> dis(0, sqrt(sigma2_n));
    // std::gamma_distribution<double> dis(4.0, 20.0);
    std::uniform_real_distribution<double> unif_dist(0.5, 20);
    // std::normal_distribution<double> temp_dist(1.0, )

    int N = 40;//100;

    VectorXd x_real = VectorXd::LinSpaced(N, 0, 1);
    VectorXd noise_vec = VectorXd::Zero(N, 1).unaryExpr([&](double dummy){return dis(gen);});
    VectorXd y_real = sin(2 * M_PI * x_real.array()).matrix() + cos(5 * M_PI * x_real.array()).matrix() + noise_vec;
    
    // plt::plot(x_real, y_real, "+r");
    // plt::plot(x_real, y_real, "o");
    // plt::grid(true);
    // plt::legend();
    // plt::show();

    /**
     * SEQUENTIAL MONTE CARLO² FOR REGRESSION PROBLEMS WITH GP KERNEL
    */

    Data sys_data;
    sys_data.N = N;
    sys_data.N_theta = 80;// 512;
    sys_data.N_x = 64;
    sys_data.Rnoise = sigma2_n;
    sys_data.B = 5000;
    sys_data.X = x_real.data();
    sys_data.Y = y_real.data();
    SMC2(sys_data);
    
}
