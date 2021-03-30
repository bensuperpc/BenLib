#!/usr/bin/env python
#
# test.py - Install for test if lib work
#
# Created by Bensuperpc at 18, July of 2020
# Modified by Bensuperpc at March 2021
#
# Released into the Private domain with MIT licence
# https://opensource.org/licenses/MIT
# MIT
#
# Written with VisualStudio code 1.4.7 and python 3.8.3
# Script compatibility : Linux (Manjaro and ArchLinux)
#
# ==============================================================================

import importlib.util
spec = importlib.util.spec_from_file_location("class_binding", "../lib/class_binding.so")
class_binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(class_binding)

from class_binding import Hello

b = Hello("Hello World OwO")
b.greet()
b.msg = "Hello World :3"
b.greet()
print(b.msg)
print("======")
c = Hello()
c.msg = "UwU"
print(c.msg)
