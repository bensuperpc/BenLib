/*
** BENSUPERPC, 2019
** 
** File description:
** >my_find_prime_sup
*/

#include "my.h"

int my_count_spe_char(const char *str, char c)
{
    int cout_char = 0;
    for (int i = 0; str[i] != '\0'; i++)
        if (str[i] == c)
            cout_char++;
    return (cout_char);
}
