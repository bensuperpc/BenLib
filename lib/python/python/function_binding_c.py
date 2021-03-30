#https://realpython.com/python-bindings-overview/

import ctypes
import pathlib

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "../lib/function_binding_c.so"
    c_lib = ctypes.CDLL(libname) 
    x, y = 6, 2.3
    c_lib.cmult.restype = ctypes.c_float
    answer = c_lib.cmult(x, ctypes.c_float(y))
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")

