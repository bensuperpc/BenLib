#!/usr/bin/env python
#
# test.py - Install for test if lib work
#
# Created by jwdinius
# Modified by Bensuperpc at 18, July of 2020
#
# Released into the Private domain with MIT licence
# https://opensource.org/licenses/MIT
# MIT
#
# Written with VisualStudio code 1.4.7 and python 3.8.3
# Script compatibility : Linux (Manjaro and ArchLinux)
#
# ==============================================================================

from pyHello import Hello
b = Hello("Hello World OwO")
b.greet()
b.msg = "Hello World :3"
b.greet()
print(b.msg)
print("======")
c = Hello()
c.msg = "UwU"
print(c.msg)
