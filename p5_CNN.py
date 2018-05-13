import os
import csv
from PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Conv2D
import matplotlib.pyplot as plt
import glob
import cv2
from keras.utils import plot_model


car_images = glob.glob('car - Copy/*.png')
cars = []
notcars = []
for image in car_images:
    cars.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))

notcar_images = glob.glob('notcar - Copy/*.png')
notcars = []
for image in notcar_images:
    notcars.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))

# Create an array stack of feature vectors
X = cars + notcars
X = np.asarray(X)
# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))
print(len(y_train))

learning_rate = 0.0005
epoch = 10
batch_size = 64

# compile and train the model using the generator function


ch, row, col = 3, 64, 64  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
#model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Conv2D(12, (5, 5), strides=(2, 2), padding='valid', activation="relu"))  # 75 320 => 36 158 64 - 30
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation="relu"))  # 36 158 => 16 77 - 13
model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='valid', activation="relu"))  # 6 37 => 4 35
model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation="relu"))  # 4 35 => 2 33
model.add(Flatten())  # 1024
model.add(Dropout(0.5))
#model.add(Dense(300, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# opt = keras.optimizers.Adam(lr=learning_rate)
opt = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = model.fit(X_train, y_train,
                           batch_size=batch_size,
                           epochs=epoch,
                           verbose=1,
                           validation_data=(X_test, y_test))

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
plot_model(model, to_file='model.png', show_shapes=True)
#model = load_model('my_model.h5')