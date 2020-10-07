/*
** BENSUPERPC, 2019
**
** File description:
** >my_compute_square_root
*/

#include "my.h"

int my_compute_square_root(int nb)
{
    int i = 0;
    int temps = 0;
    while (i < nb) {
        temps = i;
        if (nb == temps * temps)
            return (temps);
        if (nb < temps * temps && nb > (temps - 1) * (temps - 1))
            return (0);
        i++;
    }
    return (0);
}
