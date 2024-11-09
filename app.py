import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from src.preprocessing import DataPreprocessor
from src.model import create_model
from PIL import Image
import time
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #0068c9;
        border-radius: 10px;
        margin: 5px 0;
    }
    .training-metrics {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_gesture_recognition_model():
    """Load the trained model"""
    model_path = os.path.join('models', 'best_model.h5')
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

def preprocess_image(image, preprocessor):
    """Preprocess image for model prediction"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return preprocessor.preprocess_single_image(image)

def predict_gesture(image, model, preprocessor):
    """Make prediction and return gesture class and confidence"""
    processed_image = preprocess_image(image, preprocessor)
    prediction = model.predict(processed_image)[0]
    predicted_class = preprocessor.classes[np.argmax(prediction)]
    confidence = float(prediction.max())
    return predicted_class, confidence

def display_prediction(predicted_class, confidence):
    """Display prediction results with styling"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Predicted Gesture")
        st.markdown(f"<h2 style='color: #0068c9;'>{predicted_class}</h2>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Confidence")
        st.markdown(f"<h2 style='color: #0068c9;'>{confidence:.2%}</h2>", 
                   unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="confidence-bar" style="width: {confidence*100}%"></div>
        """, unsafe_allow_html=True)

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics_placeholder):
        super(TrainingCallback, self).__init__()
        self.metrics_placeholder = metrics_placeholder
        self.training_history = {
            'accuracy': [], 'val_accuracy': [],
            'loss': [], 'val_loss': []
        }

    def on_epoch_end(self, epoch, logs=None):
        for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            self.training_history[metric].append(logs.get(metric))
        
        with self.metrics_placeholder.container():
            # Display current metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Accuracy", f"{logs['accuracy']:.2%}")
            with col2:
                st.metric("Validation Accuracy", f"{logs['val_accuracy']:.2%}")
            with col3:
                st.metric("Training Loss", f"{logs['loss']:.4f}")
            with col4:
                st.metric("Validation Loss", f"{logs['val_loss']:.4f}")
            
            # Plot training history
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=self.training_history['accuracy'], 
                                   name='Training Accuracy', mode='lines'))
            fig.add_trace(go.Scatter(y=self.training_history['val_accuracy'], 
                                   name='Validation Accuracy', mode='lines'))
            fig.update_layout(title='Training Progress', 
                            xaxis_title='Epoch',
                            yaxis_title='Accuracy')
            st.plotly_chart(fig)

def train_model_ui():
    st.markdown("## Model Training 🚀")
    
    # Training parameters
    with st.expander("Training Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Number of epochs", min_value=1, value=20)
            batch_size = st.number_input("Batch size", min_value=1, value=32)
            validation_split = st.slider("Validation split", 0.1, 0.4, 0.2)
        with col2:
            learning_rate = st.number_input("Learning rate", 
                                          min_value=0.0001, 
                                          max_value=0.1, 
                                          value=0.001,
                                          format="%f")
            early_stopping_patience = st.number_input("Early stopping patience", 
                                                    min_value=1, 
                                                    value=5)
    
    if st.button("Start Training"):
        # Create progress placeholder
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        
        try:
            # Initialize preprocessor and load data
            preprocessor = DataPreprocessor('data')
            X, y = preprocessor.load_and_preprocess()
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Create and compile model
            model = create_model(input_shape=X_train.shape[1:], 
                               num_classes=len(preprocessor.classes))
            
            # Callbacks
            training_callback = TrainingCallback(metrics_placeholder)
            callbacks = [
                training_callback,
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Final evaluation
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Display final results
            st.success("Training completed successfully! 🎉")
            st.markdown("### Final Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
            with col2:
                st.metric("Test Loss", f"{test_loss:.4f}")
            
            # Save training timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['last_training'] = timestamp
            
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")
        finally:
            progress_bar.empty()

def main():
    st.title("✨ Hand Gesture Recognition System")
    st.markdown("### 👋 Prodigy InfoTech ML Internship - Task 3")
    
    # Sidebar
    st.sidebar.title("📑 Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["About", "Train Model", "Try with Image", "Real-time Detection"])
    
    if app_mode == "About":
        st.markdown("""
        ## About this App 🎯
        
        This application demonstrates real-time hand gesture recognition using deep learning. 
        
        ### Features ✨
        - Model training interface
        - Real-time gesture recognition
        - Upload and analyze images
        - Support for 10 different gestures
        
        ### Supported Gestures 👐
        1. Palm
        2. L Gesture
        3. Fist
        4. Thumb Up
        5. Index Point
        6. OK Sign
        7. Down Sign
        8. Peace Sign
        9. Stop Sign
        10. Victory Sign
        
        ### How to Use 📝
        1. Start with "Train Model" to train your model
        2. Use "Try with Image" to upload and analyze images
        3. Try "Real-time Detection" to use your webcam
        """)

    elif app_mode == "Train Model":
        train_model_ui()
        
    elif app_mode == "Try with Image" or app_mode == "Real-time Detection":
        # Load model
        model = load_gesture_recognition_model()
        if model is None:
            st.warning("⚠️ No trained model found. Please train the model first!")
            return
            
        preprocessor = DataPreprocessor('data')
        
        if app_mode == "Try with Image":
            st.markdown("## Upload an Image 📸")
            
            uploaded_file = st.file_uploader("Choose an image...", 
                                           type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze Gesture"):
                    with st.spinner("Analyzing..."):
                        predicted_class, confidence = predict_gesture(
                            image_array, model, preprocessor)
                        
                    st.success("Analysis Complete!")
                    display_prediction(predicted_class, confidence)

        else:  # Real-time Detection
            st.markdown("## Real-time Detection 📹")
            st.markdown("Click Start to begin real-time detection using your webcam.")
            
            if st.button("Start Detection"):
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                prediction_placeholder = st.empty()
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access webcam")
                            break
                        
                        predicted_class, confidence = predict_gesture(
                            frame, model, preprocessor)
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frame, channels="RGB", use_column_width=True)
                        
                        with prediction_placeholder:
                            display_prediction(predicted_class, confidence)
                        
                        time.sleep(0.1)
                        
                        if st.button("Stop Detection"):
                            break
                            
                finally:
                    cap.release()

if __name__ == "__main__":
    main()