# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:44:04 2019

@author: wmy
"""

import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

class DataLoader(object):
    
    def __init__(self, dataset_path=r'.\datasets\CombinedImages'):
        self.image_height = 256
        self.image_width = 256
        self.dataset_path = dataset_path
        pass
    
    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    def find_images(self, path):
        result = []
        for filename in os.listdir(path):
            _, ext = os.path.splitext(filename.lower())
            if ext == ".jpg" or ext == ".png":
                result.append(os.path.join(path, filename))
                pass
            pass
        result.sort()
        return result
    
    def load_data(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        batch_images = np.random.choice(search_result, size=batch_size)
        images_A = []
        images_B = []
        for image_path in batch_images:
            combined_image = self.imread(image_path)
            h, w, c = combined_image.shape
            nW = int(w/2)
            image_A, image_B = combined_image[:, :nW, :], combined_image[:, nW:, :]
            image_A = scipy.misc.imresize(image_A, (self.image_height, self.image_width))
            image_B = scipy.misc.imresize(image_B, (self.image_height, self.image_width))
            if not for_testing and np.random.random() < 0.5:
                # 数据增强，左右翻转
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)
                pass
            images_A.append(image_A)
            images_B.append(image_B)
            pass
        images_A = np.array(images_A)/127.5 - 1.
        images_B = np.array(images_B)/127.5 - 1.
        return images_A, images_B

    def load_batch(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        self.n_complete_batches = int(len(search_result) / batch_size)
        for i in range(self.n_complete_batches):
            batch = search_result[i*batch_size:(i+1)*batch_size]
            images_A, images_B = [], []
            for image_path in batch:
                combined_image = self.imread(image_path)
                h, w, c = combined_image.shape
                nW = int(w/2)
                image_A = combined_image[:, :nW, :]
                image_B = combined_image[:, nW:, :]
                image_A = scipy.misc.imresize(image_A, (self.image_width, self.image_height))
                image_B = scipy.misc.imresize(image_B, (self.image_width, self.image_height))
                if not for_testing and np.random.random() > 0.5:
                    # 数据增强，左右翻转
                    image_A = np.fliplr(image_A)
                    image_B = np.fliplr(image_B)
                    pass
                images_A.append(image_A)
                images_B.append(image_B)
                pass
            images_A = np.array(images_A)/127.5 - 1.
            images_B = np.array(images_B)/127.5 - 1.
            yield images_A, images_B
