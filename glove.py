import os
import numpy as np
import cv2
from PIL import Image 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Function to load and preprocess image data
def load_and_preprocess_data(folder_path, label, num_samples=None):
    files = os.listdir(folder_path)[:num_samples]
    data = []

    for img_file in files:
        image = Image.open(os.path.join(folder_path, img_file))
        image = image.resize((128, 128))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

    labels = [label] * len(files)
    return data, labels

# Load data for images with masks
with_gloves_path = r'C:\Users\megha\OneDrive\Desktop\mini project\datasets final\with_gloves'
with_gloves_data, with_gloves_labels = load_and_preprocess_data(with_gloves_path, label=1)

# Load data for images without masks
without_gloves_path = r'C:\Users\megha\OneDrive\Desktop\mini project\datasets final\without_gloves'
without_gloves_data, without_gloves_labels = load_and_preprocess_data(without_gloves_path, label=0)

# Combine data and labels
data = with_gloves_data + without_gloves_data
labels = with_gloves_labels + without_gloves_labels

# Convert data and labels to numpy arrays
X = np.array(data)
Y = np.array(labels, dtype=np.int)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scale the data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Define the model
num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_classes, activation='softmax'))  # Use softmax for multi-class

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)

# Plot the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# Make predictions on a new image
def predict_glove(input_image_path):
    input_image = cv2.imread(input_image_path)

    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    input_image_resized = cv2.resize(input_image, (128, 128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 1:
        result = 1
    else:
        result = 0
    
    return result