import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
paddings = tf.constant([[0,0], [2,2], [2,2], [0,0]])  # Pad 2 pixels on all sides for height and width
x_train = tf.pad(x_train, paddings, constant_values=0)
x_test = tf.pad(x_test, paddings, constant_values=0)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
x_train = tf.dtypes.cast(x_train, tf.float32) / 255.0  # Normalize pixel values
x_test = tf.dtypes.cast(x_test, tf.float32) / 255.0

model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(36, 36, 3)))  # Adjust input shape
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Softmax activation for classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Change loss function
print(model.summary())
model.fit(x_train, y_train, epochs=20)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy =', test_accuracy)

(_, _), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
x_test = tf.dtypes.cast(x_test, tf.float32) / 255.0  # Normalize pixel values

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(tf.reshape(x_test[i], [32, 32, 3]))  # Reshape for RGB images
    plt.xlabel(class_names[y_test[i][0]])  # Access the class label from y_test
plt.show()