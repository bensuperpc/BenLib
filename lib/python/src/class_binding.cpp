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
//  Source:     https://www.boost.org/doc/libs/1_75_0/libs/python/doc/html/tutorial/tutorial/exposing.html#tutorial.exposing.inheritance // CPU: ALL //
//                                                          //
//////////////////////////////////////////////////////////////

#include <boost/python.hpp>
#include <iostream>
#include <string>

class Hello {
  private:
    std::string m_msg;

  public:
    Hello()
    {
    }
    Hello(std::string msg) : m_msg(msg)
    {
    }

    void greet()
    {
        std::cout << m_msg << std::endl;
    }

    // Getter and Setter
    void set_msg(std::string msg)
    {
        this->m_msg = msg;
    }
    std::string get_msg() const
    {
        return m_msg;
    }

    bool f(int a)
    {
        return true;
    }

    bool f(int a, double b)
    {
        return true;
    }

    bool f(int a, double b, char c)
    {
        return true;
    }

    int f(int a, int b, int c)
    {
        return a + b + c;
    };
};

/// Python function 1
bool (Hello::*fx1)(int) = &Hello::f;
/// Brief description.
/** Detailed description. */
bool (Hello::*fx2)(int, double) = &Hello::f;
/**
 * OK
 *
 *@attention Not working
 *
 *@pre Context:
 * - Runtime
 * - Build
 *@post test:
 *    - FOO 
 *    - BAR
 */
/**@post poste :D. */
bool (Hello::*fx3)(int, double, char) = &Hello::f;
/**
 * A brief history of JavaDoc-style (C-style) comments.
 *
 * This is the typical JavaDoc-style C-style comment. It starts with two
 * asterisks.
 *
 * @param theory Even if there is only one possible unified theory. it is just a
 *               set of rules and equations.
 * @see publicVar()
 * @return void
 */
int (Hello::*fx4)(int, int, int) = &Hello::f;

using namespace boost::python;

BOOST_PYTHON_MODULE(class_binding)
{
    class_<Hello>("Hello")
        .def(init<std::string>())
        .def(init<>())
        .def("greet", &Hello::greet)
        .def("f", fx1)
        .def("f", fx2)
        .def("f", fx3)
        .def("f", fx4)
        .add_property("msg", &Hello::get_msg, &Hello::set_msg);
}
