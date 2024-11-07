"""
This file contains the main application for the Hand Gesture Recognition System.

The application uses the `GestureRecognitionApp` class to load the model, preprocess images, make predictions, and plot confidence scores.

The application is built using Streamlit and offers a user-friendly interface to upload an image and recognize the hand gesture.

The application also displays performance metrics from training, including the classification report.

"""
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os


class GestureRecognitionApp:
    """
    Class for the Hand Gesture Recognition Application.

    Attributes:
        model (tf.keras.Model): The loaded model.
        model_loaded (bool): Whether the model is loaded.
        gesture_mapping (dict): A dictionary mapping each gesture name to a unique integer label.
    """

    def __init__(self):
        """
        Initializes the GestureRecognitionApp class.

        Tries to load the model from the 'models' directory. If the model is not found, sets `model_loaded` to False.
        """
        try:
            self.model = tf.keras.models.load_model('models/best_model.h5')
            self.model_loaded = True
        except:
            self.model_loaded = False

        self.gesture_mapping = {
            0: 'Palm',
            1: 'L Shape',
            2: 'Fist',
            3: 'Thumb Up',
            4: 'Index Pointing',
            5: 'OK Sign',
            6: 'Down',
            7: 'Peace Sign',
            8: 'Stop Sign',
            9: 'Spider-Man'
        }

    def preprocess_image(self, image):
        """
        Preprocesses an image for prediction.

        Args:
            image (PIL.Image or numpy array): The image to preprocess.

        Returns:
            numpy array: The preprocessed image.
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = np.array(image.convert('L'))

        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = image.reshape(1, 128, 128, 1)
        return image

    def predict_gesture(self, image):
        """
        Makes a prediction for the given image.

        Args:
            image (PIL.Image or numpy array): The image to predict.

        Returns:
            tuple: A tuple containing the predicted gesture name, confidence score, and all prediction scores.
        """
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image, verbose=0)
        gesture_id = np.argmax(prediction[0])
        confidence = prediction[0][gesture_id]
        return self.gesture_mapping[gesture_id], confidence, prediction[0]

    def plot_confidence_bars(self, predictions):
        """
        Plots the confidence scores for all gestures.

        Args:
            predictions (numpy array): The prediction scores for all gestures.

        Returns:
            matplotlib figure: The plotted figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        gestures = list(self.gesture_mapping.values())
        y_pos = np.arange(len(gestures))

        # Create horizontal bar plot
        bars = ax.barh(y_pos, predictions)

        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gestures)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Confidence')
        ax.set_title('Prediction Confidence for Each Gesture')

        # Add percentage labels on the bars
        for i, v in enumerate(predictions):
            ax.text(v + 0.01, i, f'{v:.1%}', va='center')

        plt.tight_layout()
        return fig

def main():
    """
    The main application function.

    Sets up the Streamlit application, loads the model, and runs the application.
    """
    st.set_page_config(
        page_title="Hand Gesture Recognition",
        page_icon="🖐️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .upload-text {
            text-align: center;
            padding: 2rem;
            border: 2px dashed #4CAF50;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("✨ Hand Gesture Recognition System")
    st.markdown("### 🖐️ Upload an image to recognize the hand gesture!")

    app = GestureRecognitionApp()

    if not app.model_loaded:
        st.error("⚠️ Error: Model file not found. Please make sure the model is trained and saved in the 'models' directory.")
        return

    # Sidebar
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("""
    This application uses deep learning to recognize hand gestures.

    ### 🎯 Supported Gestures:
    - Palm 🖐️ 
    - L Shape 👆
    - Fist 👆
    - Fist Moved ✋
    - Thumb Up 👍
    - Index 👍
    - OK Sign 👌
    - Down 👇
    - Palm Moved 🖐️
    - C ✌️

    ### 🔧 Technical Details:
    - Model: CNN with BatchNorm
    - Input Size: 128x128
    """)

    # Performance metrics from training
    if os.path.exists('results/classification_report.txt'):
        st.sidebar.markdown("---")
        st.sidebar.title("📊 Model Performance")
        with open('results/classification_report.txt', 'r') as f:
            st.sidebar.code(f.read())

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction button
            if st.button('🔍 Recognize Gesture'):
                with st.spinner('Analyzing image...'):
                    try:
                        # Get prediction
                        gesture, confidence, all_predictions = app.predict_gesture(image)

                        # Display results
                        st.markdown("### 🎯 Results")

                        # Create success message with emoji
                        result_emoji = {
                            "Palm": "🖐️",
                            "L Shape": "👆",
                            "Fist": "✊",
                            "Thumb Up": "👍",
                            "Index": "👆",
                            "OK Sign": "👌",
                            "Down": "👇",
                            "Fist Moved": "✊",
                            "Palm Moved": "🖐️",
                            "C": "✌️"
                        }

                        st.success(f"Detected Gesture: {result_emoji.get(gesture, '')} {gesture}")
                        st.progress(float(confidence))
                        st.markdown(f"Confidence: **{confidence:.2%}**")

                        # Show confidence plot in col2
                        with col2:
                            st.markdown("### 📊 Confidence Scores")
                            fig = app.plot_confidence_bars(all_predictions)
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
        else:
            st.markdown("""
                <div class="upload-text">
                    <h3>👆 Upload an image to get started!</h3>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                </div>
                """, unsafe_allow_html=True)

            # Show sample images in col2
            with col2:
                st.markdown("### 💡 Sample Gestures")
                sample_gestures = {
                    "Palm": "🖐️",
                    "OK Sign": "👌",
                    "Thumbs Up": "👍",
                    "Fist": "✊"
                }

                for gesture, emoji in sample_gestures.items():
                    st.markdown(f"{emoji} **{gesture}**")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Task 4 - Prodigy InfoTech Machine Learning Internship</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
