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
//  file: simple_add                                        //
//  Benchmark CPU with Optimization                         //
//  Source: https://stackoverflow.com/a/26518143/10152334                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

void kernel simple_add(global const int* A, global const int* B, global int* C, 
                        global const int* N) {
    int ID, Nthreads, n, ratio, start, stop;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    n = N[0];

    ratio = (n / Nthreads);  // number of elements for each thread
    start = ratio * ID;
    stop  = ratio * (ID + 1);

    for (int i=start; i<stop; i++)
        C[i] = A[i] + B[i];
}; 
