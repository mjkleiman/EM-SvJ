import os
import pandas as pd
import numpy as np

# Set directory here
directory = 'E:/OneDrive/EM-Search/MJK-SvJ/Data'

def resize_svj(folder):
    os.chdir(directory)
    os.chdir('./' + folder)
    for f in os.listdir():
        with open(f) as g:
            gazedata = np.genfromtxt(f, dtype=np.int32, delimiter=',', skip_header=1)
            gazedata = gazedata[:,1:3]
            gazedata = np.multiply(gazedata,0.25)
            gazedata[gazedata >= 320] = 319 # Set values that are somehow above 1500 to 1500 to stop errors
            gazedata = np.int32(gazedata)
            # print(gazedata)
            np.savetxt('Small/' + f, gazedata,delimiter=',')

resize_svj("Test")