import glob
import os
import numpy as np
path = 'directory\\'
dic={}

for filename in glob.glob(os.path.join(path, '*.output')):
    key = filename
    result=np.genfromtxt(filename, delimiter=';',skip_header=3)
    print(key,end=": ")
    for entry in result:
        print(int(entry), end=",")
    print()
    dic[filename]=result