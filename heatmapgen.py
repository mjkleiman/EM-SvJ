import numpy as np
import os
import cv2
from scipy.misc import imsave

directory = 'E:/OneDrive/EM-Search/MJK-SvJ/Data/A'
os.chdir(directory)

def gen_heatmaps():

    for f in os.listdir(os.getcwd()):
        out = np.zeros([256, 256])
        with open(f) as g:
            gazedata = np.genfromtxt(f, dtype=np.int32, delimiter=',', skip_header=1)
            gazedata = gazedata[:, 1:3]
            gazedata = np.multiply(gazedata, 0.2)
            gazedata[gazedata >= 256] = 255
            xgazedata = np.int32(gazedata[:,0])
            ygazedata = np.int32(gazedata[:,1])


        for i in range(len(ygazedata)):
            out[ygazedata[i], xgazedata[i]] += 255

        out[:, :] = cv2.blur(out[:, :], (3, 3))
        imsave("Pics/" + f + '.png', out)


gen_heatmaps()