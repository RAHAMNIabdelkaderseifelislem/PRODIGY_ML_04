import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

"""
This module provides a DataPreprocessor class for loading, preprocessing, and splitting
image data for gesture recognition tasks. The images are resized, normalized, and their 
labels are converted to one-hot encoding. The class also supports data splitting into 
training, validation, and test sets.
"""

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(64, 64)):
        """
        Initialize the DataPreprocessor with a data directory and image size.

        :param data_dir: Directory containing the image data.
        :param img_size: Tuple specifying the target size for resizing images.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = self._get_classes()
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
    def _get_classes(self):
        """
        Retrieve all unique gesture classes from the first subject's directory.

        :return: A sorted list of class names.
        """
        first_subject = os.listdir(self.data_dir)[0]
        classes = [d for d in os.listdir(os.path.join(self.data_dir, first_subject)) 
                  if os.path.isdir(os.path.join(self.data_dir, first_subject, d))]
        return sorted(classes)
    
    def load_and_preprocess(self):
        """
        Load and preprocess images from the data directory.

        :return: Tuple of numpy arrays (X, y) where X are the images and y are the labels.
        """
        images = []
        labels = []
        
        # Iterate through all subjects
        for subject in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
                
            # Iterate through all gesture classes
            for gesture_class in self.classes:
                gesture_path = os.path.join(subject_path, gesture_class)
                if not os.path.isdir(gesture_path):
                    continue
                    
                # Load and preprocess images
                for img_name in os.listdir(gesture_path):
                    if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img_path = os.path.join(gesture_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    # Resize and normalize image
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(self.class_mapping[gesture_class])
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape images to include channel dimension
        X = X.reshape((-1, self.img_size[0], self.img_size[1], 1))
        
        # Convert labels to one-hot encoding
        y = to_categorical(y, num_classes=len(self.classes))
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.2):
        """
        Split the data into training, validation, and test sets.

        :param X: Numpy array of images.
        :param y: Numpy array of one-hot encoded labels.
        :param test_size: Proportion of the data to be used as the test set.
        :param validation_size: Proportion of the training data to be used as the validation set.
        :return: Tuple of numpy arrays (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_size, 
            random_state=42,
            stratify=y_train_val.argmax(axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_single_image(self, image):
        """
        Preprocess a single image for prediction.

        :param image: The input image to preprocess.
        :return: The preprocessed image ready for model prediction.
        """
        img = cv2.resize(image, self.img_size)
        img = img / 255.0
        img = img.reshape(1, self.img_size[0], self.img_size[1], 1)
        return img
