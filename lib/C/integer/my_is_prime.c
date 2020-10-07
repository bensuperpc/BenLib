/*
** BENSUPERPC, 2019
**
** File description:
** >my_is_prime
*/

#include "my.h"

int my_is_prime(int nb)
{
    if (nb == 0 || nb == 1 || nb < 0)
        return (0);
    if (nb % 2 == 0 && nb != 2)
        return (0);
    int i = 1;
    int nbrdivider = 0;
    while (nbrdivider < 3 && nb > i) {
        if (nb % i == 0)
            nbrdivider++;
        if (nbrdivider > 2 || nb == i)
            return (0);
        i++;
    }
    if (nbrdivider > 2)
        return (0);
    else
        return (1);
}
