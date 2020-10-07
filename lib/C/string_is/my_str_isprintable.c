/*
** BENSUPERPC, 2019
** 
** File description:
** >my_str_isprintable
*/

#include "my_str_isprintable.h"

int my_str_isprintable(char const *str)
{
    if (str[0] == '\0')
        return (0);
    int i = 0;
    while (str[i] != '\0') {
        if (str[i] <= 31)
            return (0);
        i++;
    }
    return (1);
}
