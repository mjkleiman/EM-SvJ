# -*- coding: utf-8 -*-

""" AlexNet.

Code adapted from tflearn example
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.

"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import os,sys,glob

directory = 'E:/OneDrive/EM-Search/MJK-SvJ/SmallPics/SvJ'


def read_ims(directory, imsz, whitening=False):
    ''' Reads in images in subdirectories located in directory and
        assigns a unique one-hot vector to each image in the respective
        folder.

        args:
             directory: the location of all the folders containing
                        each image class.
             imsz: resizes the width and height of each image to
                   imsz
             whiten: Whitens images. Default is no whitening. Images
                     must be grayscale.
             save: saves the images and labels as an h5 file. Arg is
                   a list with three strings containing the key for the
                   data and the key for the labels. For example,
                   ['images_labels.h5', 'images', 'labels'].
                   Defaults to no saving. '''

    main_dir = os.getcwd()
    os.chdir(directory)
    if whitening is True:
        num_channels = 1
    else:
        num_channels = 3
    num_ims = sum([len(files) for r, d, files in os.walk(directory)])
    imgs = np.zeros([num_ims, imsz, imsz, num_channels])
    labels = np.zeros([num_ims, len(os.listdir(os.getcwd()))])
    im_num = 0

    for f in os.listdir(os.getcwd()):
        if os.path.isdir(f):
            print('Folder name: %s' % (f))
            os.chdir(f)
            r0 = np.argmin(np.sum(labels, axis=1))
            c0 = np.argmin(np.sum(labels, axis=0))
            labels[r0:r0 + len(glob.glob1(os.getcwd(), '*')), c0] = 1

            for filename in os.listdir(os.getcwd()):
                # print(filename)
                im = imresize(imread(filename), [imsz, imsz])
                if whitening is True:
                    im = whiten(scale(im.flatten()))
                    im = im.reshape([imsz, imsz, 1])
                imgs[im_num, :, :, :] = im
                if im.shape[2] != num_channels:
                    print('Check %s file, wrong size' % (filename))
                    sys.exit(0)
                im_num += 1
            os.chdir(directory)
    os.chdir(main_dir)
    return imgs, labels


X, Y = read_imgs(directory, 227)

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_svj_all')