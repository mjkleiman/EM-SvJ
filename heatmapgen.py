import numpy as np
import os
from scipy.misc import *
import cv2

directory = 'E:/OneDrive/EM-Search/MJKSearch Coords/Raw'
os.chdir(directory)

def gen_heatmaps():
    filenum = sum([len(files) for r, d, files in os.walk(os.getcwd())])  # count number of files in directory
    output = np.zeros([filenum, 1280, 1024])
