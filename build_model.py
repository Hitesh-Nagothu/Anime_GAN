from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.models import load_model
from keras.layers import Input

from sklearn.utils import shuffle

from scipy.interpolate import make_interp_spline
from collections import deque
import matplotlib.pyplot as plt
import time
import cv2
import tqdm

import glob
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gs

import os

os.environ["KERAS_BACKEND"] = ["tensorflow"]


def get_gen_normal(noise_shape):
    kernel_init = 'glorot uniform'

    # 1
    gen_input = Input(shape=noise_shape)
    generator = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding='valid',
                                data_format="channels_last", kernel_initializer=kernel_init)(gen_input)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # 2
    generator = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # 3
    generator = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # 4
    generator = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last", kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # 5

    generator = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last",
                       kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)

    # 6
    generator = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                data_format="channels_last",
                                kernel_initializer=kernel_init)(generator)
    generator = Activation('tanh')(generator)

    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input=gen_input, output=generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model


def get_disc_normal(image_shape=(64, 64, 3)):
    image_shape = image_shape

    dropout_prob = 0.4

    kernel_init = 'glorot_uniform'

    dis_input = Input(shape=image_shape)
    # 1
    discriminator = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)

    # 2
    discriminator = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    # 3
    discriminator = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    # 4
    discriminator = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Flatten()(discriminator)

    # discriminator = MinibatchDiscrimination(100,5)(discriminator)
    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)

    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input=dis_input, output=discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model




