/*
** BENSUPERPC, 2019
** 
** File description:
** >my_str_isnum
*/

#include "my_str_isnum.h"

int my_str_isnum(char const *str)
{
    if (str[0] == '\0')
        return (0);
    int i = 0;
    while (str[i] != '\0') {
        if (str[i] <= 47 || str[i] >= 58)
            return (0);
        i++;
    }
    return (1);
}
