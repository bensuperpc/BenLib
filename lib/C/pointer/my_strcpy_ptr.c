/*
** BENSUPERPC, 2019
** 
** File description:
** >my_strcpy_ptr.c
*/

#include "my_strcpy_ptr.h"

void my_strcpy_ptr(char **dest, char const *src)
{
    unsigned long long i = 0;

    while (src[i] != '\0') {
        dest[0][i] = src[i];
        i++;
    }
    dest[0][i] = '\0';
}
