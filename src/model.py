"""
Hand Gesture Recognition Model

This module defines a convolutional neural network model for hand gesture 
recognition using TensorFlow and Keras. The model consists of multiple 
convolutional layers followed by dense layers. It also includes utility 
methods for building the model and obtaining training callbacks.

Example usage:
    model_builder = HandGestureModel()
    model = model_builder.build_model()
    model.summary()

Attributes:
    input_shape (tuple): The shape of the input images.
    num_classes (int): The number of gesture classes.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

class HandGestureModel:
    def __init__(self, input_shape=(64, 64, 1), num_classes=10):
        """
        Initializes the HandGestureModel class.

        Args:
            input_shape (tuple): The shape of the input images.
            num_classes (int): The number of gesture classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """
        Builds the CNN model for hand gesture recognition.

        Returns:
            model (tf.keras.Model): The compiled Keras model.
        """
        # Define the sequential model
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model with optimizer, loss, and metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    @staticmethod
    def get_callbacks():
        """
        Provides a list of callbacks for model training.

        Returns:
            callbacks (list): A list of Keras callback instances.
        """
        callbacks = [
            # Early stopping to stop training when a monitored metric has stopped improving
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            # Reduce learning rate when a metric has stopped improving
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            # Save the model after every epoch
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        return callbacks

if __name__ == "__main__":
    # Example usage
    model_builder = HandGestureModel()
    model = model_builder.build_model()
    model.summary()
