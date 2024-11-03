import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Experiment with different stride and filter sizes
filter_size = (5, 5)
stride_size = (1, 1)  # Reduced stride size to preserve dimensions
padding_type = "same"  # Added padding to maintain spatial dimensions

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, filter_size, strides=stride_size, padding=padding_type, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2), padding=padding_type),
    layers.Conv2D(64, filter_size, strides=stride_size, padding=padding_type, activation="relu"),
    layers.MaxPooling2D((2, 2), padding=padding_type),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model and store history for learning curves
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Plot learning curve (accuracy)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Function to plot weight and bias distributions
def plot_distribution(layer, layer_name):
    weights, biases = layer.get_weights()
    plt.hist(weights.flatten(), bins=100, alpha=0.6, color="blue", label="Weights")
    plt.hist(biases.flatten(), bins=100, alpha=0.6, color="red", label="Biases")
    plt.title(f"Distribution of Weights and Biases for {layer_name}")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Plot distributions for each Conv2D layer
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Conv2D):
        plot_distribution(layer, f"Conv2D Layer {i+1}")

import numpy as np
import matplotlib.pyplot as plt

# Predict labels for the test set
predicted_labels = model.predict(test_images)
predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert probabilities to class labels
true_labels = np.argmax(test_labels, axis=1)

# Find indices of correctly and incorrectly classified images
correct_indices = np.where(predicted_labels == true_labels)[0]
incorrect_indices = np.where(predicted_labels != true_labels)[0]

# Function to plot images
def plot_images(indices, title, n_images=10):
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    for i, idx in enumerate(indices[:n_images]):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(test_images[idx].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"Pred: {predicted_labels[idx]}\nTrue: {true_labels[idx]}")
    plt.show()

# Plot some correctly classified images
plot_images(correct_indices, "Correctly Classified Images", n_images=10)

# Plot some incorrectly classified images
plot_images(incorrect_indices, "Misclassified Images", n_images=10)

from tensorflow.keras.models import Model

# Choose an image to visualize feature maps (e.g., the first test image)
sample_image = test_images[0:1]

# Define a new model that outputs the feature maps from each convolutional layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Get feature maps for the sample image
feature_maps = activation_model.predict(sample_image)

# Function to plot feature maps
def plot_feature_maps(feature_maps, layer_name, n_columns=8):
    n_features = feature_maps.shape[-1]  # Number of feature maps in the layer
    n_rows = (n_features // n_columns) + 1
    plt.figure(figsize=(15, n_rows * 2))
    plt.suptitle(f"Feature Maps for Layer: {layer_name}")
    for i in range(n_features):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
        plt.axis("off")
    plt.show()

# Plot feature maps for each convolutional layer
for i, feature_map in enumerate(feature_maps):
    layer_name = model.layers[i].name
    plot_feature_maps(feature_map, layer_name)
