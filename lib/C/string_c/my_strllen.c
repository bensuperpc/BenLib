/*
** BENSUPERPC, 2019
**
** File description:
** >my_strllen
*/

#include "my_strllen.h"

unsigned long long my_strllen(char const *str)
{
    if (str == NULL)
        return (0);
    long long i = 0;
    while (str[i] != '\0')
        i++;
    return (i);
}
