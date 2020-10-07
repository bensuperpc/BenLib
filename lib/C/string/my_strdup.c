/*
** BENSUPERPC, 2019
** my_strdup
** File description:
** >my_strdup
*/

#include "my_strdup.h"

char *my_strdup(char const *src)
{
    char *str = NULL;
    str = (char *)malloc(sizeof(char) * (my_strllen(src) + 1));
    str = my_strcpy(str, src);

    if (str == NULL)
        return NULL;
    return str;
}
