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
    K_mat += 1e-6 * MatrixXd::Identity(n, m);
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
        }   
    }
}

// __global__ 
// void setup_curand_theta(curandStateMtgp32 *state)
// {
//     // Initialize curand with a different state for each thread
//     int idx = threadIdx.x+blockDim.x*blockIdx.x;
//     curand_init(1, idx, 0, &state[idx]);
// }

// __global__ 
// void setup_curand_x(curandStateMtgp32 *state)
// {
//     // Initialize curand with a different state for each thread
//     int idx = threadIdx.x+blockDim.x*blockIdx.x;
//     curand_init(2, idx, 0, &state[idx]);
// }

__device__
void print_matrix(const int &m, const int &n, const double *A, const int &lda) 
{
    for (int i = 0; i < m; i++) {
        printf("|");
        for (int j = 0; j < n; j++) {
            printf("%0.4f ", A[j * lda + i]);
            printf("\033[%dG", 8 * j);
        }
        printf("|\n");
    }
}