#include <vector>
#include <iostream>
#include <string>

#include "../../lib/filesystem/filesystem.hpp"

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::list_all_files(list_files,".");

    std::ios_base::sync_with_stdio(false);
    if(list_files.size() < 50)
    {
        for(const auto & elem : list_files)
        {
            std::cout << elem << std::endl;
        }
    }
    std::cout << "There is " << list_files.size() << " Files" << std::endl;

}