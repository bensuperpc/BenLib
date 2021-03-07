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
//  file: OpenCL_test.cpp                                   //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/a/26518143/10152334                                                 //
//          https://stackoverflow.com/questions/48096034/a-simple-example-with-opencl                                                 //
//			https://github.com/Sable/Ostrich/blob/master/combinational-logic/crc/opencl/crc_algo.c                                                 //
//			https://github.com/Sable/Ostrich/blob/master/combinational-logic/crc/opencl/crc_kernel.cl                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#define POLY 0xEDB88320

__kernel void CRC32_1byte_tableless(__global const uchar *data, ulong length, uint previousCrc32, __global uint *resultCrc32)
{
	__private uint crc = ~previousCrc32; // same as previousCrc32 ^ 0xFFFFFFFF

	while (length-- != 0) {
		uchar s = (uchar)crc  ^ *data++;
        uint low = (s ^ (s << 6)) & 0xFF;
		uint a = (low * ((1 << 23) + (1 << 14) + (1 << 2)));
		crc = (crc >> 8) ^ (low * ((1 << 24) + (1 << 16) + (1 << 8))) ^ a ^ (a >> 1) ^ (low * ((1 << 20) + (1 << 12))) ^ (low << 19) ^ (low << 17) ^ (low >> 2);
	}

	*resultCrc32 = ~crc;
}

__kernel void CRC32_bitwise(__global const uchar *data, ulong length, uint previousCrc32, __global uint *resultCrc32)
{
	
	__private uint crc = ~previousCrc32; // same as previousCrc32 ^ 0xFFFFFFFF
	
	while (length--) {
        crc ^= *data++;
        for (uint j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (-(int)(crc & 1) & POLY);
		}
    }
}