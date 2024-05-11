import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load CIFAR-10 dataset and split into training, validation, and testing sets
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split the full training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.4, random_state=42)

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))  # Dropout layer with 20% dropout rate
model.add(layers.Dense(10, activation='softmax'))

# Set the learning rate
learning_rate = 0.001  # You can adjust this value as needed
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model with the specified optimizer
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 20 epochs
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# Save the trained model with the correct filepath extension
model.save('E:\\5th semester\\Image Processing\\Assignment_03\\model.h5')  # Using .h5 extension for HDF5 format

# Evaluate the model on the testing dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {test_accuracy}')

# Get model predictions for the testing dataset
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
