"""
DataPreprocessor utility for the hand gesture recognition project

This utility provides methods for loading and preprocessing hand gesture
images, preparing the dataset for training, and analyzing the class distribution
of the dataset.

The prepare_dataset() method loads and preprocesses the images in the dataset,
resizes them to the specified size, normalizes the pixel values, and converts
the labels to categorical. The method returns the preprocessed dataset split
into training and testing sets.

The get_class_distribution() method calculates the class distribution of the
dataset and returns it as a pandas DataFrame.

Example usage:

    preprocessor = DataPreprocessor("data")
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset()

    # Print dataset information
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Display class distribution
    distribution = preprocessor.get_class_distribution(y_train)
    print("\nClass distribution in training set:")
    print(distribution)

"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import pandas as pd


class DataPreprocessor:
    """
    DataPreprocessor class for the hand gesture recognition project

    The class provides methods for loading and preprocessing hand gesture
    images, preparing the dataset for training, and analyzing the class
    distribution of the dataset.

    Attributes:
        data_dir (str): The path to the dataset directory.
        img_size (tuple): The size to which the images should be resized.
        gesture_mapping (dict): A dictionary mapping each gesture name to a
            unique integer label.
    """

    def __init__(self, data_dir, img_size=(64, 64)):
        """
        Initializes the DataPreprocessor class.

        Args:
            data_dir (str): The path to the dataset directory.
            img_size (tuple): The size to which the images should be resized.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.gesture_mapping = {
            '01_palm': 0,
            '02_l': 1,
            '03_fist': 2,
            '04_fist_moved': 3,
            '05_thumb': 4,
            '06_index': 5,
            '07_ok': 6,
            '08_palm_moved': 7,
            '09_c': 8,
            '10_down': 9
        }

    def load_and_preprocess_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            A numpy array representing the preprocessed image.
        """
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize image
        img = cv2.resize(img, self.img_size)
        # Normalize pixel values
        img = img / 255.0
        return img

    def prepare_dataset(self):
        """
        Prepares the dataset for training.

        Returns:
            A tuple containing the preprocessed dataset split into training
            and testing sets.
        """
        images = []
        labels = []

        # Iterate through each subject directory
        for subject in tqdm(os.listdir(self.data_dir), desc="Processing subjects"):
            subject_path = os.path.join(self.data_dir, subject)
            if os.path.isdir(subject_path):
                # Iterate through each gesture directory
                for gesture in os.listdir(subject_path):
                    if gesture in self.gesture_mapping:
                        gesture_path = os.path.join(subject_path, gesture)
                        # Process each image in the gesture directory
                        for img_name in os.listdir(gesture_path):
                            img_path = os.path.join(gesture_path, img_name)
                            img = self.load_and_preprocess_image(img_path)
                            images.append(img)
                            labels.append(self.gesture_mapping[gesture])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Reshape images to include channel dimension
        X = X.reshape(-1, self.img_size[0], self.img_size[1], 1)

        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def get_class_distribution(self, y):
        """
        Calculates the class distribution of the dataset.

        Args:
            y (numpy array): The labels of the dataset.

        Returns:
            A pandas DataFrame containing the class distribution of the dataset.
        """
        # Get class distribution for analysis
        class_counts = np.sum(y, axis=0)
        class_names = list(self.gesture_mapping.keys())
        distribution_df = pd.DataFrame({
            'Gesture': class_names,
            'Count': class_counts
        })
        return distribution_df

if __name__ == "__main__":
    """
    Example usage of the DataPreprocessor class.

    This example shows how to load and preprocess the dataset, and how to
    print the class distribution of the dataset.
    """
    # Create an instance of the DataPreprocessor class
    preprocessor = DataPreprocessor("data")

    # Prepare the dataset
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset()

    # Print dataset information
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Display class distribution
    distribution = preprocessor.get_class_distribution(y_train)
    print("\nClass distribution in training set:")
    print(distribution)
