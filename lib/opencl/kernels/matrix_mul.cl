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
//  Created: 06, March, 2021                                //
//  Modified: 06, March, 2021                               //
//  file: findStringInv.cl                                  //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/a/24160476/10152334                                                //
//          http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/docs/OpenCL_Best_Practices_Guide.pdf                                                //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

/**
 * \brief Mult 2D matrix
 * \param A 2D Matrix 1
 * \param B 2D Matrix 2
 * \param C Return result
 * \param 2D Matrix size A
 * \param 2D Matrix size B
 */

__kernel void matrixMul(__global float *C, __global float *A, __global float *B, int wA, int wB)
{
    // 2D Thread ID
    int tx = get_global_id(0);
    int ty = get_global_id(1);
    float value = 0;
    // value stores the element that is 
    // computed by the thread
    for (int k = 0; k < wA; ++k) {
        float elementA = A[ty * wA + k];
        float elementB = B[k * wB + tx];
        value += elementA * elementB;
    }
    // Write the matrix to device memory each
    // thread writes one element
    C[ty * wA + tx] = value;
}

/**
 * \brief Mult 2D matrix
 * \param A 2D Matrix 1
 * \param B 2D Matrix 2
 * \param C Return result
 * \param 2D Matrix size A
 * \param 2D Matrix size B
 */

__kernel void simpleMultiply(__global float *A, __global float *B, __global float *C, int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += A[row * TILE_DIM + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}



