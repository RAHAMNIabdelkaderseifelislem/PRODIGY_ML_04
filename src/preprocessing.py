import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(64, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' not found!")
            
        # Get and validate classes
        self.classes = self._get_classes()
        if not self.classes:
            raise ValueError("No valid gesture classes found in the data directory!")
            
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
    def _get_classes(self):
        """Get all unique gesture classes from the first subject's directory"""
        try:
            # Get all subdirectories
            subject_dirs = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
            
            if not subject_dirs:
                raise ValueError("No subject directories found!")
                
            # Get first subject directory
            first_subject = subject_dirs[0]
            first_subject_path = os.path.join(self.data_dir, first_subject)
            
            # Get gesture classes
            classes = [d for d in os.listdir(first_subject_path) 
                      if os.path.isdir(os.path.join(first_subject_path, d))]
            
            return sorted(classes)
        except Exception as e:
            raise ValueError(f"Error getting gesture classes: {str(e)}")
    
    def validate_data_structure(self):
        """Validate the data directory structure and return statistics"""
        stats = {
            'total_images': 0,
            'subjects': 0,
            'images_per_class': {},
            'errors': []
        }
        
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                stats['errors'].append(f"Data directory '{self.data_dir}' not found!")
                return stats
            
            # Count subjects
            subject_dirs = [d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))]
            stats['subjects'] = len(subject_dirs)
            
            if stats['subjects'] == 0:
                stats['errors'].append("No subject directories found!")
                return stats
            
            # Initialize image counts
            for gesture_class in self.classes:
                stats['images_per_class'][gesture_class] = 0
            
            # Count images for each class
            for subject in subject_dirs:
                subject_path = os.path.join(self.data_dir, subject)
                
                for gesture_class in self.classes:
                    gesture_path = os.path.join(subject_path, gesture_class)
                    
                    if not os.path.exists(gesture_path):
                        continue
                    
                    images = [f for f in os.listdir(gesture_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
                    stats['images_per_class'][gesture_class] += len(images)
                    stats['total_images'] += len(images)
            
            # Validate minimum requirements
            if stats['total_images'] == 0:
                stats['errors'].append("No valid images found in the dataset!")
            
            for gesture_class, count in stats['images_per_class'].items():
                if count == 0:
                    stats['errors'].append(f"No images found for class '{gesture_class}'")
                    
        except Exception as e:
            stats['errors'].append(f"Error validating data structure: {str(e)}")
            
        return stats
    
    def load_and_preprocess(self):
        """Load and preprocess all images"""
        # Validate data structure first
        stats = self.validate_data_structure()
        if stats['errors']:
            raise ValueError("\n".join(stats['errors']))
        
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
        
        if not images:
            raise ValueError("No valid images were loaded from the dataset!")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape images to include channel dimension
        X = X.reshape((-1, self.img_size[0], self.img_size[1], 1))
        
        # Convert labels to one-hot encoding
        y = to_categorical(y, num_classes=len(self.classes))
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.2):
        """Split data into train, validation and test sets"""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty dataset provided for splitting!")
            
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
        """Preprocess a single image for prediction"""
        img = cv2.resize(image, self.img_size)
        img = img / 255.0
        img = img.reshape(1, self.img_size[0], self.img_size[1], 1)
        return img