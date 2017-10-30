import numpy as np
import os
import cv2
from scipy.misc import imsave

directory = 'E:/OneDrive/EM-Search/MJK-SvJ/Data/Test'
os.chdir(directory)

def gen_heatmaps():

    for f in os.listdir(os.getcwd()):
        out = np.zeros([320, 320])
        with open(f) as g:
            gazedata = np.genfromtxt(f, dtype=np.int32, delimiter=',', skip_header=1)
            xgazedata = gazedata[:,1]
            ygazedata = gazedata[:,2]
            xgazedata = np.multiply(xgazedata,0.25)
            ygazedata = np.multiply(ygazedata, 0.25)
            xgazedata[xgazedata >= 320] = 319 # Set values that are somehow above 1500 to 1500 to stop errors
            ygazedata[ygazedata >= 320] = 319


        for i in range(len(ygazedata)):
            out[ygazedata[i], xgazedata[i]] += 255

        # out[:, :] = cv2.blur(out[:, :], (4, 4))
        imsave("Pics/" + f + '.png', out)


gen_heatmaps()