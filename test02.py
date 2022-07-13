from tabnanny import verbose
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

PATH = 'C:/Users/Amir/Desktop/final/data/'
STEP = 18*18 # 324
dim = int(np.sqrt(STEP))
NUM_SENSORS = 3
NUM_CLASS = 3 # (0, 1, 2) NAFAR
MODEL_PATH = 'C:/Users/Amir/Desktop/final/model/'

# read min_data, max_data from file
with open(MODEL_PATH+'min_max_data.txt', 'r') as f:
    min_max_str = f.readline()
min_max_str = min_max_str.split(',')
min_data = float(min_max_str[0])
max_data = float(min_max_str[1])

# load keras moodel from MODEL_PATH
model = tf.keras.models.load_model(MODEL_PATH)

model.summary()