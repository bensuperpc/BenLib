
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
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_TPP
#define MY_CUDA_MATRIX_TPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <matrix.hpp>
#include <stdio.h>

template <typename T> void my::cuda::flatten1D(const T *a, T *b, const size_t xMax, const size_t yMax)
{
//#pragma omp parallel for collapse(2) schedule(auto)
#pragma omp parallel for schedule(auto)
    for (size_t y = 0; y < yMax; y++) {
        for (size_t x = 0; x < xMax; x++) {
            b[y * xMax + x] = a[y][x];
        }
    }
}

template <typename T> void my::cuda::flatten1D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax)
{
#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t z = 0; z < zMax; z++) {
        for (size_t y = 0; y < yMax; y++) {
            for (size_t x = 0; x < xMax; x++) {
                b[(z * xMax * yMax) + (y * xMax) + x] = a[z][y][x];
            }
        }
    }
    // Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
}

template <typename T> void my::cuda::flatten1D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax)
{
#pragma omp parallel for collapse(3) schedule(auto)
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

template <typename T> void my::cuda::reshape2D(const T *a, T *b, const size_t xMax, const size_t yMax)
{
//#pragma omp parallel for collapse(2) schedule(auto)
#pragma omp parallel for schedule(auto)
    for (size_t y = 0; y < yMax; y++) {
        for (size_t x = 0; x < xMax; x++) {
            b[y][x] = a[y * xMax + x];
        }
    }
}

template <typename T> void my::cuda::reshape3D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax)
{
#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t z = 0; z < zMax; z++) {
        for (size_t y = 0; y < yMax; y++) {
            for (size_t x = 0; x < xMax; x++) {
                b[z][y][x] = a[(z * xMax * yMax) + (y * xMax) + x];
            }
        }
    }
    // Flat[x + WIDTH * (y + DEPTH * z)] = Original[x, y, z]
}

template <typename T> void my::cuda::reshape4D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax)
{
#pragma omp parallel for collapse(3) schedule(auto)
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

#endif