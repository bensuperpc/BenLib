/*
** BENSUPERPC, 2019
**
** File description:
** >my_is_prime
*/

#include "my_is_prime.h"

int my_is_prime(int num)
{
    if (num <= 1)
        return 0;
    if (num % 2 == 0 && num > 2)
        return 0;
    for (int i = 3; i < num / 2; i += 2) {
        if (num % i == 0)
            return 0;
    }
    return 1;
}
