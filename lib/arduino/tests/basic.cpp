/**
 * @file basic.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

 // g++ -D FILE_ARG=test2.ino test1.cpp -x test2.ino
// g++ -D FILE_ARG=test2.ino test1.cpp -xc++-header test2.ino
// clang++ -D FILE_ARG=test2.ino test1.cpp -xc++-header test2.ino

//https://stackoverflow.com/a/59504443/10152334
//https://stackoverflow.com/a/6671729/10152334

#define Q(x) #x
#define QUOTE(x) Q(x)

#ifdef FILE_ARG
#include QUOTE(FILE_ARG)
#endif

int main() {
    setup();
    while(1)
    {
        loop();
    }
    return 0;
}