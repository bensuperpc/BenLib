
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
//  Created: 20, March, 2021                                //
//  Modified: 20, March, 2021                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//          https://gist.github.com/AndiH/2e2f6cd9bccd64ec73c3b1d2d18284e0
//          https://stackoverflow.com/a/14038590/10152334
//          https://www.daniweb.com/programming/software-development/threads/292133/convert-1d-array-to-2d-array
//          https://stackoverflow.com/questions/20266201/3d-array-1d-flat-indexing/20266350
//          https://stackoverflow.com/a/34363345/10152334
//          https://www.gamedev.net/forums/topic/635420-4d-arrays/?page=1
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_OPS_TPP
#define MY_CUDA_MATRIX_OPS_TPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <matrix_ops.hpp>
#include <stdio.h>

template <typename T> void my::cuda::copy(T ***B_, int ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
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

template <typename T> void my::cuda::copy(T **B_, int **A_, size_t sizeX_, size_t sizeY_)
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

template <typename T> void my::cuda::display(T ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
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

template <typename T> void my::cuda::display(T **A_, size_t sizeX_, size_t sizeY_)
{
    for (size_t i = 0; i < sizeY_; i++) {
        for (size_t j = 0; j < sizeX_; j++) {
            std::cout << A_[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

template <typename T> void my::cuda::display(T *A_, size_t sizeX_)
{
    for (size_t j = 0; j < sizeX_; j++) {
        std::cout << A_[j] << " ";
    }
    std::cout << std::endl;
}

template <typename T> T ****my::cuda::aalloc(size_t sizeX_, size_t sizeY_, size_t sizeZ_, size_t sizeW_)
{
    T ****A_ = new T ***[sizeW_];
    /*
    #pragma omp parallel num_threads(2) shared(A_)
    {
        #pragma omp for nowait schedule(auto)
    */
    for (size_t i = 0; i < sizeW_; i++) {
        A_[i] = new T **[sizeZ_];
        for (size_t j = 0; j < sizeZ_; j++) {
            A_[i][j] = new T *[sizeY_];
            for (size_t k = 0; k < sizeY_; k++)
                A_[i][j][k] = new T[sizeX_];
        }
    }
    //}
    return A_;
}

template <typename T> T ***my::cuda::aalloc(size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    T ***A_ = new T **[sizeZ_];
    for (size_t i = 0; i < sizeZ_; i++) {
        A_[i] = new T *[sizeY_];
        for (size_t j = 0; j < sizeY_; j++) {
            A_[i][j] = new T[sizeX_];
        }
    }
    return A_;
}

template <typename T> T **my::cuda::aalloc(size_t sizeX_, size_t sizeY_)
{
    T **A_ = new T *[sizeY_];
    for (size_t i = 0; i < sizeY_; i++) {
        A_[i] = new T[sizeX_];
    }
    return A_;
}

template <typename T> T *my::cuda::aalloc(size_t sizeX_)
{
    T *A_ = new T[sizeX_];
    // if (A_ == NULL) { perror("malloc failure"); exit(EXIT_FAILURE); };
    return A_;
}

template <typename T> void my::cuda::adealloc(T ****A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_, size_t sizeW_)
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

template <typename T> void my::cuda::adealloc(T ***A_, size_t sizeX_, size_t sizeY_, size_t sizeZ_)
{
    for (size_t i = 0; i < sizeZ_; i++) {
        for (size_t j = 0; j < sizeY_; j++) {
            delete[] A_[i][j];
        }
        delete[] A_[i];
    }
    delete[] A_;
}

template <typename T> void my::cuda::adealloc(T **A_, size_t sizeX_, size_t sizeY_)
{
    for (size_t i = 0; i < sizeY_; i++)
        delete[] A_[i];

    delete[] A_;
}

template <typename T> void my::cuda::adealloc(T *A_, size_t sizeX_)
{
    delete[] A_;
}

template <typename T> void my::cuda::adealloc(T *A_)
{
    delete[] A_;
}

#endif