# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:32:37 2019

@author: wmy
"""

import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from model import Pix2Pix
from PIL import Image

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def predict_single_image(pix2pix, image_path, save_path):
    pix2pix.generator.load_weights('./weights/generator_weights.h5')
    image_B = imread(image_path)
    image_B = scipy.misc.imresize(image_B, (pix2pix.nW, pix2pix.nH))
    images_B = []
    images_B.append(image_B)
    images_B = np.array(images_B)/127.5 - 1.
    generates_A = pix2pix.generator.predict(images_B)
    generate_A = generates_A[0]
    generate_A = np.uint8((np.array(generate_A) * 0.5 + 0.5) * 255)
    generate_A = Image.fromarray(generate_A)
    generated_image = Image.new('RGB', (pix2pix.nW, pix2pix.nH))
    generated_image.paste(generate_A, (0, 0, pix2pix.nW, pix2pix.nH))
    generated_image.save(save_path, quality=95)
    pass

def convert_to_gray_single_image(image_path, save_path, resize_height=256, resize_weidth=256): 
    img = Image.open(image_path)
    img_color = img.resize((resize_weidth, resize_height), Image.ANTIALIAS)
    img_gray = img_color.convert('L')
    img_gray = img_gray.convert('RGB')
    img_gray.save(save_path, quality=95)
    
gan = Pix2Pix()
#gan.train(epochs=1200, batch_size=4, sample_interval=10, load_pretrained=True)

predict_single_image(gan, './images/test_1.jpg', './images/generate_test_1.jpg')
