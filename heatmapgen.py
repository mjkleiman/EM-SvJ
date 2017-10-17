import numpy as np
import os
import cv2
from PIL import Image

directory = 'E:/OneDrive/EM-Search/MJK-SvJ/Data/Test'
os.chdir(directory)

# def gen_heatmaps():

out = np.zeros([1024, 1280])

for f in os.listdir(os.getcwd()):
    with open(f) as g:
        gazedata = np.genfromtxt(f, dtype=np.int32, delimiter=',', skip_header=1)
        xgazedata = gazedata[:,1]
        ygazedata = gazedata[:,2]

    for i in range(len(ygazedata)):
        out[ygazedata[i], xgazedata[i]] += 255

    # out[:, :] = cv2.blur(out[:, :], (5, 5))
    img = Image.fromarray(out)
    img.show()
    # img.convert('RGB')
    # img.save(f, PNG, optimize=True)



