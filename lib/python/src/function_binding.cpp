//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 30, March, 2021                                //
//  Modified: 20, March, 2021                               //
//  file: function_binding.h                                //
//  python                                                  //
//  Source:                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <string>

std::string get()
{
    return "hello, world";
}

std::string set(std::string str)
{
    return str;
}

BOOST_PYTHON_MODULE(function_binding)
{
    using namespace boost::python;
    def("get", get);
    using namespace boost::python;
    def("set", set);
}
