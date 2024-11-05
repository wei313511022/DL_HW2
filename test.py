import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Optional: Call the model to ensure it's built
model.predict(train_images[:1])  # Make a prediction with a dummy input

# Create a new model that outputs the activations of each convolutional layer
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Choose an image from the test set to visualize its feature maps
test_image = test_images[0:1]  # Use the first image, reshape to match the input shape
activations = activation_model.predict(test_image)

# Function to plot feature maps
def plot_feature_maps(activations):
    for i, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        size = activation.shape[1]  # Assuming feature maps are square
        n_cols = num_filters // 8  # Number of columns in the plot
        n_rows = 8  # Fixed number of rows

        plt.figure(figsize=(n_cols, n_rows))
        for j in range(num_filters):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(activation[0, :, :, j], cmap='viridis')  # Visualize the feature map
            plt.axis('off')
        plt.suptitle(f'Feature Maps from Layer {i+1}')
        plt.show()

# Call the plotting function
plot_feature_maps(activations)
