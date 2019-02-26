# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:27:09 2019

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
from utils import DataLoader
from PIL import Image

class Pix2Pix(object):
    
    def __init__(self):
        self.nH = 256
        self.nW = 256
        self.nC = 3
        self.data_loader = DataLoader()
        self.image_shape = (self.nH, self.nW, self.nC)
        self.image_A = Input(shape=self.image_shape)
        self.image_B = Input(shape=self.image_shape)
        self.discriminator = self.creat_discriminator()
        self.discriminator.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        self.generator = self.creat_generator()
        self.fake_A = self.generator(self.image_B)
        self.discriminator.trainable = False
        self.valid = self.discriminator([self.fake_A, self.image_B])
        self.combined = Model(inputs=[self.image_A, self.image_B], outputs=[self.valid, self.fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (int(self.nH/2**4), int(self.nW/2**4), 1)
        pass
    
    def creat_generator(self):
        # layer 0
        d0 = Input(shape=self.image_shape)
        # layer 1
        d1 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        # layer 2
        d2 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        # layer 3
        d3 = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        # layer 4
        d4 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        # layer 5
        d5 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)
        d5 = BatchNormalization(momentum=0.8)(d5)
        # layer 6
        d6 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d5)
        d6 = LeakyReLU(alpha=0.2)(d6)
        d6 = BatchNormalization(momentum=0.8)(d6)
        # layer 7
        d7 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d6)
        d7 = LeakyReLU(alpha=0.2)(d7)
        d7 = BatchNormalization(momentum=0.8)(d7)
        # layer 6
        u6 = UpSampling2D(size=2)(d7)
        u6 = Conv2D(filters=512, kernel_size=4, strides=1, padding='same', activation='relu')(u6)
        u6 = BatchNormalization(momentum=0.8)(u6)
        u6 = Concatenate()([u6, d6])
        # layer 5
        u5 = UpSampling2D(size=2)(u6)
        u5 = Conv2D(filters=512, kernel_size=4, strides=1, padding='same', activation='relu')(u5)
        u5 = BatchNormalization(momentum=0.8)(u5)
        u5 = Concatenate()([u5, d5])
        # layer 4
        u4 = UpSampling2D(size=2)(u5)
        u4 = Conv2D(filters=512, kernel_size=4, strides=1, padding='same', activation='relu')(u4)
        u4 = BatchNormalization(momentum=0.8)(u4)
        u4 = Concatenate()([u4, d4])
        # layer 3
        u3 = UpSampling2D(size=2)(u4)
        u3 = Conv2D(filters=256, kernel_size=4, strides=1, padding='same', activation='relu')(u3)
        u3 = BatchNormalization(momentum=0.8)(u3)
        u3 = Concatenate()([u3, d3])
        # layer 2
        u2 = UpSampling2D(size=2)(u3)
        u2 = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu')(u2)
        u2 = BatchNormalization(momentum=0.8)(u2)
        u2 = Concatenate()([u2, d2])
        # layer 1
        u1 = UpSampling2D(size=2)(u2)
        u1 = Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu')(u1)
        u1 = BatchNormalization(momentum=0.8)(u1)
        u1 = Concatenate()([u1, d1])
        # layer 0
        u0 = UpSampling2D(size=2)(u1)
        u0 = Conv2D(self.nC, kernel_size=4, strides=1, padding='same', activation='tanh')(u0)
        return Model(d0, u0)
    
    def creat_discriminator(self):
        # layer 0
        image_A = Input(shape=self.image_shape)
        image_B = Input(shape=self.image_shape)
        combined_images = Concatenate(axis=-1)([image_A, image_B])
        # layer 1
        d1 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(combined_images)
        d1 = LeakyReLU(alpha=0.2)(d1)
        # layer 2
        d2 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        # layer 3
        d3 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        # layer 4
        d4 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model([image_A, image_B], validity)
    
    def train(self, epochs, batch_size=1, sample_interval=50, load_pretrained=False):
        if load_pretrained:
            print('Info: weights loaded.')
            self.generator.load_weights('./weights/generator_weights.h5')
            self.discriminator.load_weights('./weights/discriminator_weights.h5')
            pass
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        for epoch in range(epochs):
            for batch_i, (images_A, images_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(images_B)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([images_A, images_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, images_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # Train the generators
                g_loss = self.combined.train_on_batch([images_A, images_B], [valid, images_A])
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % 
                       (epoch+1, epochs, batch_i+1, self.data_loader.n_complete_batches, 
                        d_loss[0], 100*d_loss[1], g_loss[0]))
                # If at save interval => save generated image samples
                if (batch_i+1) % sample_interval == 0:
                    self.save_sample_images(epoch+1, batch_i+1)
                    pass
                if (batch_i+1) % 500 == 0:
                    self.generator.save_weights('./weights/generator_weights.h5')
                    self.discriminator.save_weights('./weights/discriminator_weights.h5')
                    print('Info: weights saved.')
                    pass
                pass
            if (epoch+1) % 10 == 0 :
                self.generator.save_weights('./weights/generator_weights.h5')
                self.discriminator.save_weights('./weights/discriminator_weights.h5')
                print('Info: weights saved.')
                pass
            pass
        self.generator.save_weights('./weights/generator_weights.h5')
        self.discriminator.save_weights('./weights/discriminator_weights.h5')
        print('Info: weights saved.')
        pass
    
    def save_sample_images(self, epoch, batch_i, save_dir=r'.\outputs'):
        batch_size = 3
        images_A, images_B = self.data_loader.load_data(batch_size=batch_size, for_testing=True)
        fake_A = self.generator.predict(images_B)
        generated_image = Image.new('RGB', (self.nW*3, self.nH*batch_size))
        for b in range(batch_size):
            image_A = np.uint8((np.array(images_A[b]) * 0.5 + 0.5) * 255)
            image_B = np.uint8((np.array(images_B[b]) * 0.5 + 0.5) * 255)
            image_fake_A = np.uint8((np.array(fake_A[b]) * 0.5 + 0.5) * 255)
            image_A = Image.fromarray(image_A)
            image_B = Image.fromarray(image_B)
            image_fake_A = Image.fromarray(image_fake_A)
            generated_image.paste(image_B, (0, b*self.nH, self.nW, (b+1)*self.nH))
            generated_image.paste(image_fake_A, (self.nW, b*self.nH, self.nW*2, (b+1)*self.nH))
            generated_image.paste(image_A, (self.nW*2, b*self.nH, self.nW*3, (b+1)*self.nH))
            pass
        generated_image.save(save_dir + "/G_%d_%d.jpg" % (epoch, batch_i), quality=95)
        pass
    
    pass

    
        
