#include <boost/python.hpp>
#include <iostream>
#include <string>

class Hello
{
private:
    std::string m_msg;
public:
    Hello() { }
    Hello(std::string msg):m_msg(msg) { }

    void greet() { std::cout << m_msg << std::endl; }
    
    //Getter and Setter
    void set_msg(std::string msg) { this->m_msg = msg; }
    std::string get_msg() const { return m_msg; }
};

using namespace boost::python;

BOOST_PYTHON_MODULE(pyHello)
{
    class_<Hello>("Hello")
    .def(init<std::string>())
    .def(init<>())
    .def("greet", &Hello::greet)
    .add_property("msg", &Hello::get_msg, &Hello::set_msg);
}
