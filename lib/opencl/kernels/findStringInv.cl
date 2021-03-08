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
//  Source: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html                                                //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

//#define ALPHABET "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
//#define alphabetSize 26

/**
 * \brief Generate Alphabetic sequence from size_t value, A=1, Z=27, AA = 28, AB = 29
 * \param A Alphabetic sequence index
 * \param B return array (char*)
 */


__kernel void findStringInv(__global ulong *length, __global const uchar *data, __global uint *resultCrc32)
{
	__private uint crc = 0xFFFFFFFF; // same as previousCrc32 ^ 0xFFFFFFFF
	while ((*length)-- != 0) {
		uchar s = (uchar)crc  ^ *data++;
        uint low = (s ^ (s << 6)) & 0xFF;
		uint a = (low * ((1 << 23) + (1 << 14) + (1 << 2)));
		crc = (crc >> 8) ^ (low * ((1 << 24) + (1 << 16) + (1 << 8))) ^ a ^ (a >> 1) ^ (low * ((1 << 20) + (1 << 12))) ^ (low << 19) ^ (low << 17) ^ (low >> 2);
	}

	*resultCrc32 = ~crc;
}


/*
kernel void findStringInv(__global ulong* A, __global char* B)
{
    __private char alpha[26] = {ALPHABET};
    // If *A < 27
    if (*A < 27) {
        B[0] = alpha[*A - 1];
        return;
    }
    
    // If *A > 27
    __private ulong i = 0;
    
    //barrier(CLK_LOCAL_MEM_FENCE); // Wait for others in the work-group
    while (*A > 0) {
        
        B[i] = alpha[(--*A) % alphabetSize];
        *A /= alphabetSize;
        ++i;
    }
    //const int idx = get_global_id(0);
    //C[idx] = A[idx] + B[idx];
   
}
 */