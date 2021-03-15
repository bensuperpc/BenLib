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

#define ALPHABET "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define alphabetSize 26

#define POLY 0xEDB88320

#define CHEATNBR 87

long find(const uint *value, const uint *data);
void __attribute__((overloadable)) copy(__private const uint* in, __private uint*  out);
void __attribute__((overloadable)) copy(__global char* s1, char* s2);
void findStringInv(ulong A, __private char*  B);


__constant const uint cheat_list[CHEATNBR] = {0xDE4B237D, 0xB22A28D1, 0x5A783FAE, 0xEECCEA2B, 0x42AF1E28, 0x555FC201, 0x2A845345, 0xE1EF01EA, 0x771B83FC,
    0x5BF12848, 0x44453A17, 0xFCFF1D08, 0xB69E8532, 0x8B828076, 0xDD6ED9E9, 0xA290FD8C, 0x3484B5A7, 0x43DB914E, 0xDBC0DD65, 0xD08A30FE, 0x37BF1B4E, 0xB5D40866,
    0xE63B0D99, 0x675B8945, 0x4987D5EE, 0x2E8F84E8, 0x1A9AA3D6, 0xE842F3BC, 0x0D5C6A4E, 0x74D4FCB1, 0xB01D13B8, 0x66516EBC, 0x4B137E45, 0x78520E33, 0x3A577325,
    0xD4966D59, 0x5FD1B49D, 0xA7613F99, 0x1792D871, 0xCBC579DF, 0x4FEDCCFF, 0x44B34866, 0x2EF877DB, 0x2781E797, 0x2BC1A045, 0xB2AFE368, 0xFA8DD45B, 0x8DED75BD,
    0x1A5526BC, 0xA48A770B, 0xB07D3B32, 0x80C1E54B, 0x5DAD0087, 0x7F80B950, 0x6C0FA650, 0xF46F2FA4, 0x70164385, 0x885D0B50, 0x151BDCB3, 0xADFA640A, 0xE57F96CE,
    0x040CF761, 0xE1B33EB9, 0xFEDA77F7, 0x8CA870DD, 0x9A629401, 0xF53EF5A5, 0xF2AA0C1D, 0xF36345A8, 0x8990D5E1, 0xB7013B1B, 0xCAEC94EE, 0x31F0C3CC, 0xB3B3E72A,
    0xC25CDBFF, 0xD5CF4EFF, 0x680416B1, 0xCF5FDA18, 0xF01286E9, 0xA841CC0A, 0x31EA09CF, 0xE958788A, 0x02C83A7C, 0xE49C3ED4, 0x171BA8CC, 0x86988DAE, 0x2BDD2FA1};


long find(const uint* value, const uint *data)
{
    size_t i = get_global_id(0);
    for(; i < CHEATNBR; i++)
    {
      if (cheat_list[i] == value[0]) {
        return i;
      }
    }
    return  -1;
}

void __attribute__((overloadable)) copy(__private const uint* in, __private uint* out)
{
  size_t i = get_global_id(0);
  out[i] = in[i];
}

void __attribute__((overloadable)) copy(__global char* s1, char* s2)
{
  //barrier(CLK_LOCAL_MEM_FENCE);
  size_t i = get_global_id(0);
  s1[i] = s2[i];
}


void findStringInv(ulong A, __private char* B)
{
  //barrier(CLK_LOCAL_MEM_FENCE);
  __private char alpha[26] = {ALPHABET};
  
  // If A < 27
  if (A < 27) {
      B[0] = alpha[A - 1];
      return;
  }
  // If A > 27
  __private ulong i = 0;
  while (A > 0) {
    B[i] = alpha[(--A) % alphabetSize];
    A /= alphabetSize;
    i++;
  }
}


kernel void findStringInv_T(__global ulong* A, __global char* B)
{
  ulong nbrs = *A;
  char str[6];
  findStringInv(nbrs, str);

  copy(B, str);

  /*
  long index = 0;
  uint value[1] = {0x5A783FAE};
  uint cheat_list_[87];

  uint cheat_list1[2] = {1, 2};
  uint cheat_list2[2] = {0, 0};

  copy(cheat_list1, cheat_list2);

  long i = find(value, cheat_list_);
  B[0] = '0' + cheat_list2[0];
  B[1] = '0' + cheat_list2[1];
  */
}