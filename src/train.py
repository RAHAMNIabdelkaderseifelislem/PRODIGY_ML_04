"""
Utility for training the hand gesture recognition model.

This script provides a simple interface for training the hand gesture
recognition model using the prepared dataset. The script uses the
DataPreprocessor class to load and preprocess the dataset, and the
HandGestureModel class to build and train the model. The script also
includes methods for plotting the training history and confusion matrix.

Attributes:
    data_dir (str): The path to the dataset directory.

Methods:
    train_model(epochs=50, batch_size=32): Trains the model using the
        prepared dataset.

    plot_training_history(history): Generates and saves a plot of the
        training history.

    plot_confusion_matrix(y_test, y_pred): Generates and saves a plot of
        the confusion matrix.
"""

import os
import numpy as np
import tensorflow as tf
from data_preprocessing import DataPreprocessor
from model import HandGestureModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


class ModelTrainer:
    """
    Class for training the hand gesture recognition model.
    """

    def __init__(self, data_dir):
        """
        Initializes the ModelTrainer class.

        Args:
            data_dir (str): The path to the dataset directory.
        """
        self.data_dir = data_dir
        self.preprocessor = DataPreprocessor(data_dir)
        self.model_builder = HandGestureModel()

    def train_model(self, epochs=50, batch_size=32):
        """
        Trains the model using the prepared dataset.

        Args:
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size to use during training.

        Returns:
            model (tf.keras.Model): The trained model.
            history (dict): The training history.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_dataset()

        # Build model
        model = self.model_builder.build_model()
        callbacks = self.model_builder.get_callbacks()

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_accuracy:.4f}")

        # Generate and save training plots
        self.plot_training_history(history)

        # Generate confusion matrix
        y_pred = model.predict(X_test)
        self.plot_confusion_matrix(y_test, y_pred)

        return model, history

    def plot_training_history(self, history):
        """
        Generates and saves a plot of the training history.

        Args:
            history (dict): The training history.
        """
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Generates and saves a plot of the confusion matrix.

        Args:
            y_test (numpy array): The true labels.
            y_pred (numpy array): The predicted labels.
        """
        # Convert one-hot encoded labels back to class indices
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()


if __name__ == "__main__":
    trainer = ModelTrainer("data")
    model, history = trainer.train_model()

    # Save the trained model
    model.save('models/final_model.h5')
