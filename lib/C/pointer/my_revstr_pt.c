/*
** BENSUPERPC, 2019
**
** File description:
** >my_revstr_pt
*/

#include "my_revstr_pt.h"

void my_revstr_pt(char **str)
{
    char temps = '0';
    long long x = 0;
    long long y = my_strllen(str[0]) - 1;

    while (x < y) {
        temps = str[0][x];
        str[0][x] = str[0][y];
        str[0][y] = temps;
        x++;
        y--;
    }
}
