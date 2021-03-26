
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
//  Created: 16, March, 2021                                //
//  Modified: 17, March, 2021                               //
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
//          https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
//          https://stackoverflow.com/a/49435122/10152334
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_TPP
#define MY_CUDA_MATRIX_TPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <matrix.hpp>
extern "C"
{
#include <stdio.h>
}

template <class... Args> void print(Args... args);

template <class... Args> void print(Args... args)
{
    (std::cout << ... << args) << "\n";
}
// print(1, ':', " Hello", ',', " ", "World!");

template <typename T> void my::cuda::flatten1D(T **a, T *b, const size_t xMax, const size_t yMax)
{
#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t y = 0; y < yMax; y++) {
        for (size_t x = 0; x < xMax; x++) {
            b[y * xMax + x] = a[y][x];
        }
    }
}

template <typename T> void my::cuda::flatten1D(T ***a, T *b, const size_t xMax, const size_t yMax, const size_t zMax)
{
#pragma omp parallel for collapse(3) schedule(auto)
    for (size_t z = 0; z < zMax; z++) {
        for (size_t y = 0; y < yMax; y++) {
            for (size_t x = 0; x < xMax; x++) {
                b[x + y * xMax + z * xMax * yMax] = a[z][y][x];
                // b[x + WIDTH * (y + DEPTH * z)]
                // Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
            }
        }
    }
    // Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
}

template <typename T> void my::cuda::flatten1D(T ****a, T *b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax)
{
#pragma omp parallel for collapse(4) schedule(auto)
    for (size_t w = 0; w < wMax; w++) {
        for (size_t z = 0; z < zMax; z++) {
            for (size_t y = 0; y < yMax; y++) {
                for (size_t x = 0; x < xMax; x++) {
                    b[x + y * xMax + z * yMax * xMax + w * yMax * xMax * zMax] = a[w][z][y][x];
                }
            }
        }
    }
    // x + y*width + z*height*width + w*height*width*depth.
}

template <typename T> void my::cuda::reshape2D(const T *a, T **b, const size_t xMax, const size_t yMax)
{
#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t y = 0; y < yMax; y++) {
        for (size_t x = 0; x < xMax; x++) {
            b[y][x] = a[y * xMax + x];
        }
    }
}

template <typename T> void my::cuda::reshape3D(const T *a, T ***b, const size_t xMax, const size_t yMax, const size_t zMax)
{
#pragma omp parallel for collapse(3) schedule(auto)
    for (size_t z = 0; z < zMax; z++) {
        for (size_t y = 0; y < yMax; y++) {
            for (size_t x = 0; x < xMax; x++) {
                b[z][y][x] = a[(z * xMax * yMax) + (y * xMax) + x];
            }
        }
    }
    // Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
}

template <typename T> void my::cuda::reshape4D(const T *a, T ****b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax)
{
#pragma omp parallel for collapse(3) schedule(auto) shared(b)
    for (size_t w = 0; w < wMax; w++) {
        for (size_t z = 0; z < zMax; z++) {
            for (size_t y = 0; y < yMax; y++) {
                for (size_t x = 0; x < xMax; x++) {
                    b[w][z][y][x] = a[x + y * xMax + z * yMax * xMax + w * yMax * xMax * zMax];
                }
            }
        }
    }
    // x + y*width + z*height*width + w*height*width*depth.
}

// 2D flat matrix
template <typename T> void my::cuda::matMultFlat(T *matA, T *matB, T *matC, const size_t m)
{
    my::cuda::matMultFlat(matA, m, m, matB, m, m, matC, m, m);
}

// 2D flat matrix
template <typename T>
void my::cuda::matMultFlat(T *matA, size_t sizeAX, size_t sizeAY, T *matB, size_t sizeBX, size_t sizeBY, T *matC, size_t sizeCX, size_t sizeCY)
{
    T tmp;
#pragma omp parallel for collapse(2) schedule(auto) private(tmp)
    for (size_t y = 0; y < sizeAX; y++) {
        for (size_t x = 0; x < sizeBY; x++) {
            for (size_t s = 0; s < sizeBX; s++) {
                tmp += matA[y * sizeAY + s] * matB[s * sizeBY + x];
            }
            matC[y * sizeCY + x] = tmp;
            tmp = (T)0.0;
        }
    }
}

// 3D flat matrix
template <typename T>
void my::cuda::matMultFlat(T *matA, size_t sizeAX, size_t sizeAY, size_t sizeAZ, T *matB, size_t sizeBX, size_t sizeBY, size_t sizeBZ, T *matC, size_t sizeCX,
    size_t sizeCY, size_t sizeCZ)
{
#warning "To do: my::cuda::matMultFlat : 3D flat matrix "
    T tmp;
#pragma omp parallel for collapse(2) schedule(auto) private(tmp)
    // The first group loop
    for (size_t z = 0; z < sizeAX; ++z) {
        for (size_t y = 0; y < sizeBY; ++y) {
            for (size_t x = 0; x < sizeBZ; ++x) {
                // The second group loop
                for (size_t s = 0; s < sizeBX; ++s) {
                    // matC[y * sizeCY + x] = matC[y * sizeCY + x] + matA[y * sizeAY + s + z * sizeAX * sizeAZ] * matB[s * sizeBY + x + z * sizeBX * sizeBZ];
                }
                // End of the second group loop
            }
        }
    }
}

template <typename T> void my::cuda::matMult(T **matA, T **matB, T ****matC, size_t sizeAX, size_t sizeAY, size_t sizeBX, size_t sizeBY)
{
    my::cuda::matMult(matA, sizeAX, sizeAY, matB, sizeBX, sizeBY, matC);
}

template <typename T> void my::cuda::matMult(T **A_, T **B_, T **C_, size_t sizeAX, size_t sizeAY, size_t sizeBX, size_t sizeBY)
{
#pragma omp parallel for collapse(3) schedule(auto)
    for (size_t y = 0; y < sizeAX; ++y) {
        for (size_t x = 0; x < sizeBY; ++x) {
            for (size_t s = 0; s < sizeBX; ++s) {
                C_[y][x] = C_[y][x] + A_[y][s] * B_[s][x];
            }
        }
    }
}

template <typename T>
void my::cuda::matMult(T ***matA, T ***matB, T ****matC, size_t sizeAX, size_t sizeAY, size_t sizeAZ, size_t sizeBX, size_t sizeBY, size_t sizeBZ)
{
    my::cuda::matMult(matA, sizeAX, sizeAY, sizeAZ, matB, sizeBX, sizeBY, sizeBZ, matC);
}

template <typename T>
void my::cuda::matMult(T ***matA, size_t sizeAX, size_t sizeAY, size_t sizeAZ, T ***matB, size_t sizeBX, size_t sizeBY, size_t sizeBZ, T ***matC)
{
#pragma omp parallel for collapse(4) schedule(auto)
    // The first group loop
    for (size_t z = 0; z < sizeAX; ++z) {
        for (size_t y = 0; y < sizeBY; ++y) {
            for (size_t x = 0; x < sizeBZ; ++x) {
                // The second group loop
                for (size_t s = 0; s < sizeBX; ++s) {
                    matC[z][y][x] = matC[z][y][x] + matA[z][y][s] * matB[s][y][x];
                }
            }
        }
    }
}

template <typename T>
void my::cuda::matMult(
    T ****matA, T ****matB, T ****matC, size_t sizeAX, size_t sizeAY, size_t sizeAZ, size_t sizeAW, size_t sizeBX, size_t sizeBY, size_t sizeBZ, size_t sizeBW)
{
    my::cuda::matMult(matA, sizeAX, sizeAY, sizeAZ, sizeAW, matB, sizeBX, sizeBY, sizeBZ, sizeBW, matC);
}

template <typename T>
void my::cuda::matMult(
    T ****matA, size_t sizeAX, size_t sizeAY, size_t sizeAZ, size_t sizeAW, T ****matB, size_t sizeBX, size_t sizeBY, size_t sizeBZ, size_t sizeBW, T ****matC)
{
#pragma omp parallel for collapse(4) schedule(auto)
    // The first group loop
    for (size_t w = 0; w < sizeAX; ++w) {
        for (size_t z = 0; z < sizeBY; ++z) {
            for (size_t y = 0; y < sizeBZ; ++y) {
                for (size_t x = 0; x < sizeBW; ++x) {
                    // The second group loop
                    for (size_t s = 0; s < sizeBX; ++s) {
                        matC[w][z][y][x] = matC[w][z][y][x] + matA[w][z][y][s] * matB[s][z][y][x];
                    }
                    // End of the second group loop
                }
            }
        }
    }
}

#endif