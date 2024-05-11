import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

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

# Get the predicted class label
predicted_class = np.argmax(predictions)
print("Predicted Class:", predicted_class)
