import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Experiment with different stride and filter sizes
filter_size = 5
stride_size = 2  # Reduced stride size to preserve dimensions
padding_type = "same"  # Added padding to maintain spatial dimensions

# Set the L2 regularization factor
l2_lambda = 0.0

# Residual block
def residual_block(x, filters, kernel_size, stride, conv_shortcut=False):
    shortcut = x
    if conv_shortcut or stride > 1:  # Use a 1x1 convolution for shortcut connection
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding=padding_type)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # First layer of the residual block
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding=padding_type,
                      kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second layer of the residual block
    x = layers.Conv2D(filters, kernel_size, padding=padding_type,
                      kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)

    # Combine the shortcut and the main path
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Define the ResNet model
def build_resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, filter_size,strides=stride_size, padding=padding_type, kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, 16, filter_size, stride_size)  # First stack of residual blocks
    x = residual_block(x, 16, filter_size, stride_size)

    x = residual_block(x, 32, filter_size, 2, conv_shortcut=True)  # Down-sample
    x = residual_block(x, 32, filter_size, stride_size)

    x = residual_block(x, 64, filter_size, 2, conv_shortcut=True)  # Down-sample
    x = residual_block(x, 64, filter_size, stride_size)

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

# Compile the model
model = build_resnet()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Custom callback to track test accuracy after each epoch
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        self.test_accuracies.append(test_acc)
        print(f"\nEpoch {epoch+1}: Test accuracy = {test_acc:.4f}")

test_accuracy_callback = TestAccuracyCallback(test_data=(test_images, test_labels))

# Train the model and store history for learning curves
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1, callbacks=[test_accuracy_callback])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# # Plot learning curve (accuracy)
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.plot(test_accuracy_callback.test_accuracies, label="Test Accuracy")
# plt.title("Learning Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# # Plot training and validation loss
# plt.plot(history.history["loss"], label="Train Loss")
# plt.title("Loss Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # Function to plot weight and bias distributions
# def plot_distribution(layer, layer_name):
#     weights, biases = layer.get_weights()
#     plt.hist(weights.flatten(), bins=100, alpha=0.6, color="blue", label="Weights")
#     plt.hist(biases.flatten(), bins=100, alpha=0.6, color="red", label="Biases")
#     plt.title(f"Distribution of Weights and Biases for {layer_name}")
#     plt.xlabel("Values")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()

# # Plot distributions for each Conv2D layer
# j = 0
# for i, layer in enumerate(model.layers):
#     if isinstance(layer, layers.Conv2D):
#         plot_distribution(layer, f"Histogram conv{j+1}")
#         j += 1

# plot_distribution(model.layers[-1], "Histogram output")

# # Predict labels for the test set
# predicted_labels = model.predict(test_images)
# predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert probabilities to class labels
# true_labels = np.argmax(test_labels, axis=1)

# # Find indices of correctly and incorrectly classified images
# correct_indices = np.where(predicted_labels == true_labels)[0]
# incorrect_indices = np.where(predicted_labels != true_labels)[0]

# # Function to plot images
# def plot_images(predict, indices, title, n_images=10):
#     plt.figure(figsize=(12, 4))
#     plt.suptitle(title)
#     j = 0
#     for i, idx in enumerate(indices):
#         if j == n_images - 1:
#             break
#         if predicted_labels[idx] == predict:
#             plt.subplot(1, n_images, j + 1)
#             plt.imshow(test_images[idx])
#             plt.axis("off")
#             plt.title(f"Pred: {predicted_labels[idx]}\nTrue: {true_labels[idx]}")
#             j += 1
#     plt.show()

# # Plot some correctly classified and misclassified images
# plot_images(5, correct_indices, "Correctly Classified Images", n_images=10)
# plot_images(5, incorrect_indices, "Misclassified Images", n_images=10)

# # Create a new model that outputs the activations of each convolutional layer
# layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
# activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

# # Choose an image from the test set
# test_image = test_images[6:7]  # Use one image, reshape to match the input shape
# activations = activation_model.predict(test_image)

# # Function to plot feature maps
# def plot_feature_maps(activations):
#     for i, activation in enumerate(activations):
#         num_filters = activation.shape[-1]
#         n_cols = 12  # Number of columns in the plot
#         n_rows = 10  # Fixed number of rows

#         plt.figure(figsize=(n_cols, n_rows))
#         for j in range(num_filters):
#             plt.subplot(n_rows, n_cols, j + 1)
#             plt.imshow(activation[0, :, :, j], cmap='viridis')  # Visualize the feature map
#             plt.axis('off')
#         plt.suptitle(f'Feature Maps from Convolution Layer {i+1}')
#         plt.show()

# # Call the plotting function
# plot_feature_maps(activations)
