#ifndef MY_REVSTR_H
#define MY_REVSTR_H
#include <string.h>
#include "my_strlen.h"

#define XOR_SWAP(a, b)                                                                                                                                         \
    do {                                                                                                                                                       \
        a ^= b;                                                                                                                                                \
        b ^= a;                                                                                                                                                \
        a ^= b;                                                                                                                                                \
    } while (0)
void my_revstr(char *);
#endif
