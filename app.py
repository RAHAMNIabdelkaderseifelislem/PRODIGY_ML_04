import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from src.preprocessing import DataPreprocessor
from PIL import Image
import time

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
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_gesture_recognition_model():
    """Load the trained model"""
    model_path = os.path.join('models', 'best_model.h5')
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train the model first.")
        return None
    return load_model(model_path)

def preprocess_image(image, preprocessor):
    """Preprocess image for model prediction"""
    # Convert to grayscale if needed
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
    
    # Display confidence bar
    st.markdown(f"""
        <div class="confidence-bar" style="width: {confidence*100}%"></div>
        """, unsafe_allow_html=True)

def main():
    st.title("✨ Hand Gesture Recognition System")
    st.markdown("### 👋 Prodigy InfoTech ML Internship - Task 3")
    
    # Initialize preprocessor and model
    preprocessor = DataPreprocessor('data')
    model = load_gesture_recognition_model()
    
    if model is None:
        return

    # Sidebar
    st.sidebar.title("📑 Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["About", "Try with Image", "Real-time Detection"])

    if app_mode == "About":
        st.markdown("""
        ## About this App 🎯
        
        This application demonstrates real-time hand gesture recognition using deep learning. 
        
        ### Features ✨
        - Real-time gesture recognition using webcam
        - Upload and analyze images
        - Support for 10 different gestures
        - High accuracy predictions
        
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
        1. Choose "Try with Image" to upload and analyze an image
        2. Choose "Real-time Detection" to use your webcam
        3. Follow the on-screen instructions
        """)

    elif app_mode == "Try with Image":
        st.markdown("## Upload an Image 📸")
        
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Analyze Gesture"):
                with st.spinner("Analyzing..."):
                    predicted_class, confidence = predict_gesture(
                        image_array, model, preprocessor)
                    
                st.success("Analysis Complete!")
                display_prediction(predicted_class, confidence)

    elif app_mode == "Real-time Detection":
        st.markdown("## Real-time Detection 📹")
        st.markdown("Click Start to begin real-time detection using your webcam.")
        
        if st.button("Start Detection"):
            # Start webcam
            cap = cv2.VideoCapture(0)
            
            # Create placeholder for webcam feed
            stframe = st.empty()
            
            # Create placeholder for predictions
            prediction_placeholder = st.empty()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                    
                    # Make prediction
                    predicted_class, confidence = predict_gesture(
                        frame, model, preprocessor)
                    
                    # Display frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame, channels="RGB", use_column_width=True)
                    
                    # Display prediction
                    with prediction_placeholder:
                        display_prediction(predicted_class, confidence)
                    
                    # Add small delay to prevent overwhelming the app
                    time.sleep(0.1)
                    
                    # Check if user wants to stop
                    if st.button("Stop Detection"):
                        break
                        
            finally:
                cap.release()

if __name__ == "__main__":
    main()
