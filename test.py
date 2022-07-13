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
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# random generating test data
# x = np.random.uniform(low=min_data, high=max_data, size=(STEP, NUM_SENSORS))

# load random sample x from file for testing model

def read_file(file_name):
    data = []
    Data = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row_data = line.split(' ')
            Data.append([int(x) for x in row_data])
    Data = np.array(Data)

    for i in range(STEP, len(Data)+1):
        x = Data[i-STEP: i]
        data.append(x)
    data = np.array(data)
    return data
data = read_file(PATH+'1/01.txt')
x = data[np.random.permutation(len(data))[0]]

# normalize x between -1, 1
x = (((x - min_data) / (max_data - min_data)) * 2.0) - 1.0

# Reshape x to (1, STEP, NUM_SENSORS)
print(len(x))
x = np.reshape(x, (1, dim, dim, NUM_SENSORS))

prediction_probability = probability_model.predict(x, verbose=0)
print('prediction_probability: ', prediction_probability)
prediction = np.argmax(prediction_probability)
print('Class: ', prediction)




