/*
** BENSUPERPC, 2019
** 
** File description:
** >my_inv_two_str_pt
*/

#include "my_inv_two_str_pt.h"

void my_inv_two_str_pt(char **str1, char **str2)
{
    char *temps = my_strdup(str1[0]);
    free(str1[0]);
    str1[0] = my_strdup(str2[0]);
    free(str2[0]);
    str2[0] = my_strdup(temps);
}
