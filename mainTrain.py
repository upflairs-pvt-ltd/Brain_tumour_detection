import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten



from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# Set the working directory to the location of 'dataset/' directory
os.chdir('/home/ranjit/Desktop/Tumor_Detection')

image_directory = 'Dataset/'

# Check if 'no/' directory exists
if os.path.exists(os.path.join(image_directory, 'no')):
    no_tumor_image = os.listdir(os.path.join(image_directory, 'no'))
else:
    print("'no/' directory does not exist.")
    exit()

yes_tumor_image = os.listdir(os.path.join(image_directory, 'yes'))
dataset = []
label = []
IMPUT_SIZE=64

for i, image_name in enumerate(no_tumor_image):
    if image_name.split('.')[1] == 'jpg':
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((IMPUT_SIZE, IMPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_image):
    if image_name.split('.')[1] == 'jpg':
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((IMPUT_SIZE, IMPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

#print(len(dataset))
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

#Model Building

model=Sequential()

model.add(Conv2D(32, (3,3),input_shape=(IMPUT_SIZE, IMPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

# Save the model
model.save('BrainTumor10Epochs.h5')

# Now, let's make predictions and print the results

# Make predictions on the test set
predictions = model.predict(x_test)

# Convert predictions to binary values (0 or 1) using a threshold (e.g., 0.5)
binary_predictions = (predictions > 0.5).astype(int)

result = model.evaluate(x_test,y_test)
print("Performance of your model :",result)



