//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 17, March, 2021                                //
//  Modified: 17, March, 2021                               //
//  file: OpenCL_test.cpp                                   //
//  Crypto                                                  //
//  Source: https://www.techiedelight.com/dynamic-memory-allocation-in-c-for-2d-3d-array/
//          http://www.cplusplus.com/forum/general/263317/
//          https://stackoverflow.com/questions/18273370/the-correct-way-to-initialize-a-dynamic-pointer-to-a-multidimensional-array
//          https://data-flair.training/blogs/multi-dimensional-arrays-in-c-cpp/
//          https://www.geeksforgeeks.org/c-program-multiply-two-matrices/
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix.hpp"
#include "matrix.tpp"
extern "C"
{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

#if defined(_OPENMP)
#    include <omp.h>
#endif

#define sizeZ 2
#define sizeY 3
#define sizeX 4

#define Min 0
#define Max 9

void copy(int ***B_, int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_);
void copy(int ***B_, int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
#if defined(_OPENMP)
#    pragma omp parallel
    {
#    pragma omp for collapse(2) schedule(auto)
#endif
        for (size_t i = 0; i < sizeZ_; i++) {
            for (size_t j = 0; j < sizeY_; j++) {
                for (size_t k = 0; k < sizeX_; k++) {
                    B_[i][j][k] = A_[i][j][k];
                }
            }
        }
#if defined(_OPENMP)
    }
#endif
}

void copy(int **B_, int **A_, size_t sizeX_, size_t sizeY_);
void copy(int **B_, int **A_, size_t sizeX_, size_t sizeY_)
{
#if defined(_OPENMP)
#    pragma omp parallel
    {
#    pragma omp for collapse(1) schedule(auto)
#endif
        for (size_t j = 0; j < sizeY_; j++) {
            for (size_t k = 0; k < sizeX_; k++) {
                B_[j][k] = A_[j][k];
            }
        }
#if defined(_OPENMP)
    }
#endif
}

void display(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_);
void display(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    for (size_t i = 0; i < sizeZ_; i++) {
        std::cout << "Depth: " << i << "\n";
        for (size_t j = 0; j < sizeY_; j++) {
            for (size_t k = 0; k < sizeX_; k++)
                std::cout << A_[i][j][k] << " ";

            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void display(int **A_, size_t sizeX_, size_t sizeY_);
void display(int **A_, size_t sizeX_, size_t sizeY_)
{
    for (size_t i = 0; i < sizeY_; i++) {
        for (size_t j = 0; j < sizeX_; j++) {
            std::cout << A_[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void display(int *A_, size_t sizeX_);
void display(int *A_, size_t sizeX_)
{
    for (size_t j = 0; j < sizeX_; j++) {
        std::cout << A_[j] << " ";
    }
    std::cout << std::endl;
}

int ****aalloc(size_t sizeX_, size_t sizeY_, size_t sizeZ_, size_t sizeW_)
{
    int ****A_ = new int ***[sizeW_];
    /*
    #pragma omp parallel num_threads(2) shared(A_)
    {
        #pragma omp for nowait schedule(auto)
    */
    for (size_t i = 0; i < sizeW_; i++) {
        A_[i] = new int **[sizeZ_];
        for (size_t j = 0; j < sizeZ_; j++) {
            A_[i][j] = new int *[sizeY_];
            for (size_t k = 0; k < sizeY_; k++)
                A_[i][j][k] = new int[sizeX_];
        }
    }
    //}
    return A_;
}

int ***aalloc(size_t sizeX_, size_t sizeY_, size_t sizeZ_);
int ***aalloc(size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    int ***A_ = new int **[sizeZ_];
    for (size_t i = 0; i < sizeZ_; i++) {
        A_[i] = new int *[sizeY_];
        for (size_t j = 0; j < sizeY_; j++) {
            A_[i][j] = new int[sizeX_];
        }
    }
    return A_;
}

int **aalloc(size_t sizeX_, size_t sizeY_);
int **aalloc(size_t sizeX_, size_t sizeY_)
{
    int **A_ = new int *[sizeY_];
    for (size_t i = 0; i < sizeY_; i++) {
        A_[i] = new int[sizeX_];
    }
    return A_;
}

int *aalloc(size_t sizeX_);
int *aalloc(size_t sizeX_)
{
    int *A_ = new int[sizeX_];
    // if (A_ == NULL) { perror("malloc failure"); exit(EXIT_FAILURE); };
    return A_;
}

void adealloc(int ****A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_, size_t sizeW_);
void adealloc(int ****A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_, size_t sizeW_)
{
    for (size_t i = 0; i < sizeW_; i++) {
        for (size_t j = 0; j < sizeZ_; j++) {
            for (size_t k = 0; k < sizeY_; k++) {
            }
            delete[] A_[i][j];
        }
        delete[] A_[i];
    }
    delete[] A_;
}

void adealloc(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_);
void adealloc(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    for (size_t i = 0; i < sizeZ_; i++) {
        for (size_t j = 0; j < sizeY_; j++) {
            delete[] A_[i][j];
        }
        delete[] A_[i];
    }
    delete[] A_;
}

void adealloc(int **A_, size_t sizeX_, size_t sizeY_);
void adealloc(int **A_, size_t sizeX_, size_t sizeY_)
{
    for (size_t i = 0; i < sizeY_; i++)
        delete[] A_[i];

    delete[] A_;
}

void adealloc(int *A_, size_t sizeX_);
void adealloc(int *A_, size_t sizeX_)
{
    delete[] A_;
}

void adealloc(int *A_);
void adealloc(int *A_)
{
    delete[] A_;
}

void matmult(int **A_, int **B_, int **C_, size_t sizeAX, size_t sizeAY, size_t sizeBX, size_t sizeBY);
void matmult(int **A_, int **B_, int **C_, size_t sizeAX, size_t sizeAY, size_t sizeBX, size_t sizeBY)
{
#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t i = 0; i < sizeAX; ++i) {
        for (size_t j = 0; j < sizeBY; ++j) {
            for (size_t k = 0; k < sizeBX; ++k) {
                C_[i][j] += A_[i][k] * B_[k][j];
            }
        }
    }
}

void multiply(int **mat1, int **mat2, int **res);
void multiply(int **mat1, int **mat2, int **res)
{
    size_t i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            res[i][j] = 0;
            for (k = 0; k < 3; k++)
                res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
}

void fillRand(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_);
void fillRand(int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    unsigned int seed;
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(auto) private(seed)
        for (size_t i = 0; i < sizeZ_; i++) {
            for (size_t j = 0; j < sizeY_; j++) {
                seed = 25234U + 16U * (unsigned int)omp_get_thread_num(); // 17U
                for (size_t k = 0; k < sizeX_; k++) {
                    A_[i][j][k] = (rand_r(&seed) % (Max - Min + 1)) + Min;
                }
            }
        }
    }
}

void fillRand(int **A_, size_t sizeX_, size_t sizeY_);
void fillRand(int **A_, size_t sizeX_, size_t sizeY_)
{
    unsigned int seed;
#pragma omp parallel
    {
#pragma omp for collapse(1) schedule(auto) private(seed)
        for (size_t j = 0; j < sizeY_; j++) {
            seed = 25234U + 16U * (unsigned int)omp_get_thread_num(); // 17U
            for (size_t k = 0; k < sizeX_; k++) {
                A_[j][k] = (rand_r(&seed) % (Max - Min + 1)) + Min;
            }
        }
    }
}

int main()
{
    /*
        int ****D = aalloc(175, 175, 175, 175);
        adealloc(D, 175, 175, 175, 175);
    */
    int **M1 = aalloc(3, 3);
    fillRand(M1, sizeX, sizeY);
    int **M2 = aalloc(3, 3);
    copy(M2, M1, sizeX, sizeY);
    int **M3 = aalloc(3, 3);
    matmult(M1, M2, M3, 3, 3, 3, 3);
    // multiply(M1, M2, M3);
    display(M1, 3, 3);
    display(M2, 3, 3);
    display(M3, 3, 3);

    auto M4 = aalloc(3 * 3);
    auto M5 = aalloc(3 * 3);
    auto M6 = aalloc(3 * 3);

    my::cuda::flatten1D<int>(M1, M4, (size_t)3, (size_t)3);

    my::cuda::flatten1D<int>(M2, M5, (size_t)3, (size_t)3);

    my::cuda::cpu_matrix_mult<int>(M4, M5, M6, 3);
    // my::cuda::cpu_matrix_mult<int>(M4, 3 ,3 ,M5, 3, 3, M6, 3, 3);

    display(M4, 3 * 3);
    display(M5, 3 * 3);
    display(M6, 3 * 3);

    auto M7 = aalloc(175 * 175 * 175 * 175);
    // memset(M7, 0, sizeof(M7) * 175*175*175*175);
    return 0;

    int ***A = aalloc(sizeX, sizeY, sizeZ);
    int ***B = aalloc(sizeX, sizeY, sizeZ);
    int ***C = aalloc(sizeX, sizeY, sizeZ);

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < sizeZ; i++) {
            A[i] = new int *[sizeY];
            B[i] = new int *[sizeY];
            C[i] = new int *[sizeY];
            for (int j = 0; j < sizeY; j++) {
                A[i][j] = new int[sizeX];
                B[i][j] = new int[sizeX];
                C[i][j] = new int[sizeX];
            }
        }
    }

    fillRand(A, sizeX, sizeY, sizeZ);

    copy(B, A, sizeX, sizeY, sizeZ);

    display(A, sizeX, sizeY, sizeZ);

    adealloc(A, sizeX, sizeY, sizeZ);
    adealloc(B, sizeX, sizeY, sizeZ);
    adealloc(C, sizeX, sizeY, sizeZ);

    return EXIT_SUCCESS;
}
