import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')

# Load and preprocess the image
img = cv2.imread('/home/ranjit/Desktop/Tumor_Detection/Dataset/yes/y1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
img = cv2.resize(img, (64, 64))  # Resize the image to match the model's input size
img = img / 255.0  # Normalize pixel values to the range [0, 1]
input_img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(input_img)


# Now, let's make predictions and print the results

# Convert predictions to binary values (0 or 1) using a threshold (e.g., 0.5)
binary_predictions = (predictions > 0.5).astype(int).ravel()
prob  =float(binary_predictions[0])
if binary_predictions[0]==1:
    print(f"Person Has Tumor and probality {prob:20f}")

else:
    print(f"congratulation you are fit, {prob:20f}")
