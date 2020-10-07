/*
** BENSUPERPC, 2019
** 
** File description:
**
*/

#include "my.h"

void my_sort_int_array(int *array, int size)
{
    int x = 0;
    int y = 0;
    while (x < size) {
        while (y < size) {
            ext_fonc_sort_ar(&array[x], &array[y]);
            y++;
        }
        y = 0;
        x++;
    }
}

void ext_fonc_sort_ar(int *a, int *b)
{
    int temps = 0;
    if (*b >= *a) {
        temps = *b;
        *b = *a;
        *a = temps;
    }
}
