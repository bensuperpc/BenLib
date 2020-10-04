
#if __cplusplus >= 201703L
#    include "printer.hpp"

namespace thread::printer::detail
{
std::mutex mutex = std::mutex();

void print()
{
}

void print(char character)
{
    std::putchar(character);
}

void print(const char *string)
{
    std::fputs(string, stdout);
}

void print(const std::string &string)
{
    std::fputs(string.c_str(), stdout);
}
} // namespace thread::printer::detail
#endif
