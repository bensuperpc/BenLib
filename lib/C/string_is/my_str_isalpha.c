/*
** BENSUPERPC, 2019
**
** File description:
** >my_str_isalpha
*/

#include "my_str_isalpha.h"

int my_str_isalpha(char const *str)
{
    if (str[0] == '\0')
        return (1);
    int i = 0;
    while (str[i] != '\0') {
        if (str[i] <= '@' || str[i] >= '{')
            return (0);
        if (str[i] >= '[' && str[i] <= '`')
            return (0);
        i++;
    }
    return (1);
}
