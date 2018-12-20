from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# limit amount of GPU Memory to use
# tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.67)))

num_classes = 10

################################################
# Preprocessing. Leave this section unchanged! #
################################################

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

##################
# Implement here #
##################

# If not stated otherwise, all activation functions have to be Rectified Linear Units

# Convolution layer with 32 output filters, a kernel size of 3x3
model.add(Conv2D(32, (3,3), input_shape=input_shape, activation='relu'))

# Convolution layer with 64 output filters, a kernel size of 3x3
model.add(Conv2D(64, (3,3), activation='relu'))

# Maxpooling layer with a pool size of 2x2
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout layer with a drop fraction of 0.5
model.add(Dropout(0.5))

# Flatten layer
model.add(Flatten())

# Fully-connected layer with 128 neurons
model.add(Dense(128, activation='relu'))

# Dropout layer with a drop fraction of 0.5
model.add(Dropout(0.5))

# Fully-connected layer with as many neurons as there are classes in the problem (Output layer), activation function: Softmax
model.add(Dense(num_classes, activation='softmax'))

# HyperParameters
# + Batch size: 128
# + Epochs: 3
# + Loss: Categorical Crossentropy
# + Optimizer: Adam
# + Learning rate: 0.001
# + Evaluation Metric: Accuracy

# Adam-Optimizer
adam = optimizers.adam(lr = 0.001)
# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit Model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6, batch_size=128)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test)
print("CNN Error: %.4f%%" % (100-scores[1]*100))
print("Accuracy: %.4f%%" % scores[1])
print("Loss: %.4f%%" % scores[0])

#load test image, flatten and normalize image data
image_file = 'digit.png'
print("\nLoading test image '%s'" % image_file)
img = image.load_img(image_file, target_size=(img_rows, img_cols), color_mode = 'grayscale')
x = image.img_to_array(img)
x /= 255.0
x = np.expand_dims(x, axis=0)

# predict class from given image
classes = model.predict_classes([x])
print("Predicted class => ", classes[0])
print("Probabilies: ", model.predict([x])[0])