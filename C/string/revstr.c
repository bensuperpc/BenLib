#include <string.h>
#include "revstr.h"

void revstr(char *str)
{
    if (str) {
        char *end = str + strlen(str) - 1;
        while (str < end) {
            XOR_SWAP(*str, *end);
            str++;
            end--;
        }
    }
}
