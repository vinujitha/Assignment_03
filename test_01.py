import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("E:\\5th semester\\Image Processing\\Assignment_03\\model.h5")

# Define a function to preprocess input images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))  # Resize to match model's input shape
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Example usage:
input_image_path = "E:\\5th semester\\Image Processing\\Assignment_03\\dog.jpg"
input_image = preprocess_image(input_image_path)

# Make predictions using the loaded model
predictions = model.predict(input_image)

# Get the predicted class label and corresponding class name
predicted_class = np.argmax(predictions)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class_name = class_names[predicted_class]

# Load and plot the input image
input_img = image.load_img(input_image_path)
plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title('Input Image')
plt.axis('off')

# Plot the predicted class label
plt.subplot(1, 2, 2)
plt.barh(class_names, predictions.reshape(-1))
plt.xlabel('Probability')
plt.title('Predicted Class: ' + predicted_class_name)
plt.tight_layout()

# Show the plot
plt.show()
