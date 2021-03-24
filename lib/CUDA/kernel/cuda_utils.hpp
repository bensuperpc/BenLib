
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
//  Created: 24, March, 2021                                //
//  Modified: 24, March, 2021                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source:
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_UTILES_HPP
#define MY_CUDA_UTILES_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
extern "C"
{
#include "stdio.h"
}

namespace my
{
namespace cuda
{
void driver();
void device();
} // namespace cuda
} // namespace my

#endif