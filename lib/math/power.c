/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** power.c
*/

#include "power.h"

long int power(long int nb, long int p)
{
    if (p < 0)
        return (0);
    if (p != 0)
        return (nb * power(nb, p - 1));
    else
        return 1;
}

bool isPowerOfTwo(int x)
{
    return x && (!(x & (x - 1)));
}