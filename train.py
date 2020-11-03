import os
import glob

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio

from PIL import Image
import matplotlib.gridspec as gs

from keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Dropout, Input
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU

from build_model import get_disc_normal, get_gen_normal

import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import make_interp_spline

K.set_image_dim_ordering('tf')

from collections import deque

np.random.seed(1337)


def norm_img(img):
    img = (img / 127.5) - 1
    return img


def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def sample_from_dataset(batch_size, image_shape, data_dir=None, data=None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)

    all_data_dirlist = list(glob.glob(data_dir))

    sample_imgs_paths = np.random.choice(all_data_dirlist, batch_size)

    for index, img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB')

        image = np.asarray(image)
        image = norm_img(image)
        sample[index, ...] = image

    return sample


def gen_noise(batch_size, noise_shape):
    return np.random.normal(0, 1, size=(batch_size,) + noise_shape)

def generate_images(generator, save_dir):
    
