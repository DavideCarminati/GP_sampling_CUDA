#ifndef PT_SAMPLER
#define PT_SAMPLER
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include "utils.hpp"
#include "cu_utils.hpp"

using namespace Eigen;

std::vector<MatrixXd> PTsampler(Data &data, Distribution &prior, Distribution &likelihood, Distribution &proposal, PToptions opts);

#endif