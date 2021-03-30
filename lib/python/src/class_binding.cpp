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

BOOST_PYTHON_MODULE(class_binding)
{
    class_<Hello>("Hello")
    .def(init<std::string>())
    .def(init<>())
    .def("greet", &Hello::greet)
    .add_property("msg", &Hello::get_msg, &Hello::set_msg);
}
