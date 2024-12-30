//code1
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

from keras.datasets import cifar10
(train_img, train_lab), (test_img, test_lab) = cifar10.load_data()


//code2
class_names = ['airplane', 'automobile', 'bird', 'dog', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))

for i in range(9):
  plt.xticks([])
  plt.yticks([])
  plt.subplot(3,3,i+1)
  plt.imshow(train_img[i])
  plt.xlabel(class_names[train_lab[i][0]])

plt.show()


//code3
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

//code4
model.add(Flatten())
model.add(Dense(100, activation='relu'))

# Adding output layer
model.add(Dense(10, activation='softmax'))

//code5
# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

//code6
# fitting the model
history = model.fit(train_img, train_lab, epochs=10, validation_data=(test_img, test_lab))

model.save('cifar.bs')