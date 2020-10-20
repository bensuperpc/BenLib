/*
** BENSUPERPC, 2019
**
** File description:
** >my_swap
*/

#include "my_swap.h"

void my_swap(int *a, int *b)
{
    *a = *a ^ *b;
    *b = *a ^ *b;
    *a = *a ^ *b;
    /*
    int *temps = 0;
    temps = a;
    a = b;
    b = temps;
    */
}
