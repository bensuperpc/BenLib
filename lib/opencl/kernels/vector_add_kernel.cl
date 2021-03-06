//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2020                                            //
//  Created: 6, March, 2021                                 //
//  Modified: 6, March, 2021                                //
//  file: vecadd.cpp                                        //
//  Benchmark CPU with Optimization                         //
//  Source: https://stackoverflow.com/a/26518143/10152334                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

kernel void vecadd( global int* A, global int* B, global int* C ) {
    const int idx = get_global_id(0);
    C[idx] = A[idx] + B[idx];
}
/*
kernel void vecadd_v2( global int* A, global int* B, global int* C ) {
    const size_t gid = get_global_id(0);
    const size_t groupSize = get_global_size(0);
    for(int i = gid; i< maxSize; i+= groupSize){
        result[i] = a[i] + b[i];
    }
}
*/
