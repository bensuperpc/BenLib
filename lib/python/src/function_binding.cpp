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

BOOST_PYTHON_MODULE(hello_ext)
{
    using namespace boost::python;
    def("get", get);
    using namespace boost::python;
    def("set", set);
}
