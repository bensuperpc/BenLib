#!/bin/bash
time find . -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec clang-format -style=file -i {} \;
#time find . -iname *.hpp -o -iname *.h -o -iname *.c | xargs clang-format -style=file -i

#clang-format --verbose -i -style=file *.c
