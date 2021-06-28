#!/usr/bin/env python3
import os
import re
from posix import listdir

path = "./data/FN/bamboo_2/clean/color_full" #TODO


#os.chdir(path)
#print(os.listdir("."))

files = listdir(path)
for file in files:
    split = file.split("_")
    file_new = str(split[0]+"_00"+split[1])
    #print(file_new)
    os.rename(os.path.join(path, file), os.path.join(path, file_new))