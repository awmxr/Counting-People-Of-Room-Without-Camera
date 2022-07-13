import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

PATH = 'C:/Users/Amir/Desktop/final/data/'
STEP = 18*18
NUM_SENSORS = 3
NUM_CLASS = 3 # (0, 1, 2) NAFAR
EPOCHS = 200
BATCH_SIZE = 1024
LEARNING_RATE=1e-5
TRAIN_PERCENTAGE = 0.8
MODEL_PATH = 'C:/Users/Amir/Desktop/final/model4/'

if os.path.exists(MODEL_PATH) == False:
    os.mkdir(MODEL_PATH)

# classes = np.array([int(class_name) for class_name in os.listdir(path)])
# NUM_CLASS = len(classes)

# -----------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------
data = []
labels = []
for class_name in os.listdir(PATH):
    label = int(class_name)
    files = [PATH+class_name+'/'+ file_name for file_name in os.listdir(PATH+class_name+'/') if file_name.lower().endswith('.txt')]
    for file_name in files:
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
            labels.append(label)
data = np.array(data)
labels = np.array(labels)

# -----------------------------------------------------------------------
# shuffle data
# -----------------------------------------------------------------------

idx = np.random.permutation(len(data))
data = data[idx]
labels = labels[idx]

# -----------------------------------------------------------------------
# normalize data between -1, 1
# -----------------------------------------------------------------------

min_data = np.min(data)
max_data = np.max(data)

# save min_data and max_data to file
with open(MODEL_PATH+'min_max_data.txt', 'w') as f:
    f.write(str(min_data)+','+str(max_data))


data = ((data - min_data) / (max_data - min_data) * 2.0 )-1.0

dim = int(np.sqrt(STEP))

# Reshape data
data = np.reshape(data, (len(data), dim , dim, NUM_SENSORS))


# -----------------------------------------------------------------------
# split data to train and test
# -----------------------------------------------------------------------

cnt_train = int(len(data) * TRAIN_PERCENTAGE)

train_data = data[:cnt_train]
train_labels = labels[:cnt_train]

test_data = data[cnt_train:]
test_labels = labels[cnt_train:]

print('train_data:',train_data.shape,'\n', 'train_labels:', train_labels.shape, '\n','test_data:', test_data.shape,'\n' ,'test_labels:', test_labels.shape)

# -----------------------------------------------------------------------
# plot mean of train data
# -----------------------------------------------------------------------
#plt.figure(figsize=(10,5 * NUM_CLASS))
for c in range(NUM_CLASS):
    plt.subplot(2, NUM_CLASS, c+1)
    idx_class = np.where(train_labels == c)[0]
    data_class = train_data[idx_class]
    mean_data_class = np.mean(data_class, axis=0)
    mean_data_class = (mean_data_class - np.min(mean_data_class)) / (np.max(mean_data_class) - np.min(mean_data_class))
    plt.imshow(mean_data_class)
    plt.title('mean: '+str(c))
    plt.axis('off')
    plt.subplot(2, NUM_CLASS, c+NUM_CLASS+1)
    std_data_class = np.std(data_class, axis=0)
    std_data_class = (std_data_class - np.min(std_data_class)) / (np.max(std_data_class) - np.min(std_data_class))
    plt.imshow(std_data_class)
    plt.title('std: '+str(c))
    plt.axis('off')
plt.show()

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# for c in range(NUM_CLASS):
#     idx_class = np.where(train_labels == c)[0]
#     data_class = train_data[idx_class]
#     x = np.reshape(data_class[:,:,0], (-1,))
#     y = np.reshape(data_class[:,:,1], (-1,))
#     z = np.reshape(data_class[:,:,2], (-1,))
#     cl = 'green' if c == 0 else 'red' if c == 1 else 'blue'
#     ax.scatter3D(x, y, z, color = cl)
# plt.title("simple 3D scatter plot")
# plt.show()

# -----------------------------------------------------------------------
# Build Model
# -----------------------------------------------------------------------

model = Sequential([
    Input(shape=(dim, dim, NUM_SENSORS)),

    Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    Flatten(),

    # Dense(256),
    # BatchNormalization(),
    # Activation('relu'),

    # Dense(128, kernel_reqularizer=tf.keras.regularizers.l2(0.01)),
    # BatchNormalization(),
    # Activation('relu'),

    #Dropout(0.5),

    Dense(128, kernel_regularizer=tf.keras.regularizers.L1(0.01)),
    BatchNormalization(),
    Activation('relu'),

    Dense(64, kernel_regularizer=tf.keras.regularizers.L1(0.01)),
    BatchNormalization(),
    Activation('relu'),

    Dropout(0.5),

    Dense(32, kernel_regularizer=tf.keras.regularizers.L1(0.01)),
    BatchNormalization(),
    Activation('relu'),

    #Dropout(0.5),

    Dense(NUM_CLASS)
])

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])

# -----------------------------------------------------------------------
# Save model structure as json format
# -----------------------------------------------------------------------

with open(MODEL_PATH+'model_conv.json', 'w') as f:
    f.write(model.to_json())

# -----------------------------------------------------------------------
# Train Model
# -----------------------------------------------------------------------

model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_data, test_labels))

# -----------------------------------------------------------------------
# Evaluate Model
# -----------------------------------------------------------------------

model.evaluate(test_data, test_labels)

# -----------------------------------------------------------------------
# Save Model
# -----------------------------------------------------------------------
model.save(MODEL_PATH)
