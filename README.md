# Mood-Based-Recommendation-System
README: Emotion Detection & Music Recommendation

Overview
This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to detect facial emotions from images or a webcam feed, and recommends a Spotify playlist based on the most frequently detected emotion.

By analyzing a user’s facial expressions, the application classifies emotions into one of seven categories:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

Once an emotion is detected, the application retrieves a Spotify playlist URL that matches the user’s emotional state.

Key Features
1. CNN Model Architecture:
   - Convolutional layers with Conv2D
   - Pooling layers with MaxPooling2D
   - Dropout layers to prevent overfitting
   - Dense layers for final classification

2. Data Loading and Preprocessing:
   - FER2013-style images (48x48, grayscale)
   - Automatic resizing and normalization with ImageDataGenerator

3. Real-Time Emotion Detection:
   - Uses OpenCV’s CascadeClassifier to detect faces in a webcam feed.
   - Feeds each detected face into the trained CNN model for emotion prediction.
   - Tracks the most frequently detected emotion over a set duration (default 30 seconds).

4. Emotion-to-Playlist Mapping:
   - A dictionary maps each emotion to a unique Spotify playlist URL.
   - When the application ends, it prints the most common emotion and its associated playlist.

File Descriptions

1. Model Creation and Training
   - CNN Model: Defined using Keras’ Sequential API with multiple convolutional, pooling, and dropout layers.
   - Compilation: Uses adam optimizer, categorical_crossentropy loss, and tracks accuracy.
   - Data Loaders (ImageDataGenerator): Automatically resizes and normalizes images from the specified folders.
   - Training: Runs for a specified number of epochs, validates on a test dataset.
   - Model Saving: Saves the trained model to emotion_detection_model.h5.

2. Model Inference with Static Images
   - Loads a single grayscale image (48×48) for testing.
   - Prints the predicted emotion label.

3. Real-Time Detection & Playlist Recommendation
   - OpenCV: Captures frames from a webcam.
   - Face Detection: Locates faces using OpenCV’s Haarcascade model (haarcascade_frontalface_default.xml).
   - Emotion Prediction: For each face, crops and resizes the region of interest, then applies the trained CNN model.
   - Aggregation: Keeps track of all detected emotions in a session.
   - Output: After 30 seconds or upon pressing q, it outputs the most common emotion and the suggested Spotify playlist link.

Requirements
- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib (optional, if you want to visualize training results)
- Spotify (for accessing the playlists, though you only need a browser link)

Example installation (pip):
pip install tensorflow opencv-python numpy matplotlib

Usage

1. Dataset Setup
   - Obtain the FER2013 (or another) dataset and structure it into:
     archive (2)/
     ├── train/
     │   ├── angry/
     │   ├── disgust/
     │   ├── fear/
     │   ├── happy/
     │   ├── neutral/
     │   ├── sad/
     │   └── surprise/
     └── test/
         ├── angry/
         ├── disgust/
         ├── fear/
         ├── happy/
         ├── neutral/
         ├── sad/
         └── surprise/
   - Ensure the train_dir and test_dir paths match your folder structure.

2. Model Training
   - Run the script up to the model.fit() call.
   - Adjust hyperparameters (epochs, batch size, etc.) as needed.
   - Once training is complete, the model will be saved as emotion_detection_model.h5.

3. Evaluation
   - Evaluate the model on the test dataset:
     loss, accuracy = model.evaluate(test_generator)
     print(f"Test Loss: {loss}")
     print(f"Test Accuracy: {accuracy}")

4. Static Image Test
   - Load and predict on a single image (e.g., div.jpg).
   - Confirm the emotion prediction.

5. Real-Time Video Detection
   - Ensure your webcam is accessible.
   - Run the section of code that initializes the webcam (cv2.VideoCapture(0)), detects faces, and predicts emotions.
   - After 30 seconds or upon pressing q, it outputs the most common emotion and the suggested Spotify playlist link.

Customization
- Duration: Change the time window (default 30 seconds) in the while loop.
- Playlists: Update the Spotify playlists in emotion_to_playlist to suit your preferences.
- Model Architecture: Modify or add more layers (e.g., additional Conv2D layers, different kernel sizes, etc.).
- Hyperparameters: Adjust the learning rate, optimizer, and dropout probabilities to improve performance.

Troubleshooting
1. OpenCV Haarcascade Path
   - Make sure the path to haarcascade_frontalface_default.xml is correct. By default, it’s loaded from cv2.data.haarcascades.

2. Webcam Issues
   - If your webcam index is different, change cv2.VideoCapture(0) to a different index or a video file path.

3. Performance / Accuracy
   - If accuracy is low, increase the number of training epochs or apply more data augmentation techniques (ImageDataGenerator has many parameters).
   - For more robust performance, consider transfer learning (using a pretrained model).

License
This project is not under an official license by default. You can add an open-source license (e.g., MIT) if you plan on sharing.

Acknowledgments
- The FER2013 dataset for providing training images.
- TensorFlow/Keras for the deep learning framework.
- OpenCV for real-time face detection.
- Spotify for playlist links.

Enjoy experimenting with the Emotion Detection & Music Recommendation project!
Feel free to reach out for any questions or suggestions.

