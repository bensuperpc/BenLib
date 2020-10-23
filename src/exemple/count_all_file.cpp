#include <iostream>
#include <string>
#include <vector>
#include "../../lib/filesystem/filesystem.hpp"

int main()
{
    std::cout << "There is " << my::filesystem::count_files(".") << " Files" << std::endl;
}