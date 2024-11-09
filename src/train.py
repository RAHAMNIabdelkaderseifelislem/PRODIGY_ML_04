"""
This file contains functions to train and evaluate a convolutional neural network (CNN) model for hand gesture recognition.

The main function is train_model, which takes a data directory as input and trains a model on the data. The model is saved to a folder called 'models' in the current directory. The function also generates and saves plots of the model's training history and confusion matrix.

The model is defined in the model.py file and is a CNN with three convolutional blocks followed by two dense layers. The model is compiled with the Adam optimizer and categorical cross-entropy loss.

The data is preprocessed using the DataPreprocessor class, which is defined in the preprocess.py file. The class loads the data, resizes the images, normalizes the pixel values, and converts the labels to one-hot encoding.

The training history and confusion matrix are plotted using the plot_training_history and plot_confusion_matrix functions, respectively. The plots are saved to the 'models' folder.

"""

import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

from preprocessing import DataPreprocessor
from model import create_model


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history and save the figure

    Args:
        history: History object returned by model.fit()
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix and save the figure

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(data_dir, model_save_path='models'):
    """Train a model on the data in the given directory

    Args:
        data_dir: Directory containing the image data
        model_save_path: Path to save the model

    Returns:
        model: Trained model
        preprocessor: DataPreprocessor object used to preprocess the data
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_dir)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = preprocessor.load_and_preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Create and compile model
    print("Creating model...")
    model = create_model(input_shape=X_train.shape[1:], num_classes=len(preprocessor.classes))
    
    # Create callbacks
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate and save plots
    plot_training_history(history, os.path.join(model_save_path, 'training_history.png'))
    
    # Generate predictions for confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, preprocessor.classes,
                         os.path.join(model_save_path, 'confusion_matrix.png'))
    
    return model, preprocessor


if __name__ == '__main__':
    train_model('data')

