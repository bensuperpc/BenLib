import importlib.util
spec = importlib.util.spec_from_file_location("function_binding", "../lib/function_binding.so")
function_binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(function_binding)

#import function_binding
#from ..lib import function_binding

print(function_binding.get())
print(function_binding.set("Bonjour :D"))
