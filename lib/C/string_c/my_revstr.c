#include "my_revstr.h"

void my_revstr(char *str)
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
