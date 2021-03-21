
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
//  Created: 21, March, 2021                                //
//  Modified: 21, March, 2021                               //
//  file: kernel.cuh                                        //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_CRC_CUH
#define MY_CUDA_CRC_CUH

#define POLY 0xEDB88320

using uchar = unsigned char;
using ulong = unsigned long;

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void JAMCRC_byte_tableless(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);

__global__ void CRC32_byte_tableless2(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);

#endif