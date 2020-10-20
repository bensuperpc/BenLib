/*
** BENSUPERPC, 2019
**
** File description:
** >my_putchar
*/

#include "my_putchar.h"

int my_putchar(char c)
{
    return (write(1, &c, 1));
}
