/*
** BENSUPERPC, 2019
** 
** File description:
** >my_putstr
*/

#include "my_putstr.h"

void my_putstr(char const *str)
{
    if (str == NULL)
        return;
    int i = 0;
    while (str[i] != '\0') {
        my_putchar(str[i]);
        i++;
    }
}
