
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
//  Source:                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_CRC_H
#define MY_CUDA_CRC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

void CRC32_byte_tableless(dim3 &gridSize, dim3 &blockSize, unsigned char *data, ulong length, uint previousCrc32, uint *resultCrc32);
void JAMCRC_byte_tableless(dim3 &gridSize, dim3 &blockSize, unsigned char *data, ulong length, uint previousCrc32, uint *resultCrc32);

void CRC32_byte_tableless2(dim3 &gridSize, dim3 &blockSize, unsigned char *data, ulong length, uint previousCrc32, uint *resultCrc32);
void JAMCRC_byte_tableless2(dim3 &gridSize, dim3 &blockSize, unsigned char *data, ulong length, uint previousCrc32, uint *resultCrc32);

#endif