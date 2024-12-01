import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher  # Import to check word similarity

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Load the trained model using TensorFlow
model = tf.keras.models.load_model('./cnn_model.h5')

# Define the labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

# Expected number of features
expected_num_features = 84

# Array to store predicted characters
predicted_word = []
last_prediction = None
repeat_count = 0
repeat_threshold = 5

# Threshold for stopping when no hand is detected
no_hand_frames_threshold = 10
no_hand_frames = 0

# Final word to be displayed
final_word = ""

# Function to load common words from file
def load_common_words(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to reduce repeated characters in the predicted word
def filter_repeated_characters(predicted_word):
    filtered_word = []
    last_char = None
    for char in predicted_word:
        if char != last_char:
            filtered_word.append(char)
        last_char = char
    return ''.join(filtered_word)

# Function to find the closest match for the predicted word from common words
def find_closest_word(predicted_word, common_words):
    max_similarity = 0
    best_match = ""
    for word in common_words:
        similarity = SequenceMatcher(None, predicted_word, word).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = word
    return best_match

# Load common words from file
common_words = load_common_words('./common_words.txt')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ''

    if results.multi_hand_landmarks:
        no_hand_frames = 0
        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                all_landmarks.append((x, y))

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = [coord for landmark in all_landmarks for coord in landmark]
        data_aux = data_aux[:expected_num_features] + [0.0] * (expected_num_features - len(data_aux))

        data_aux = np.array(data_aux).reshape(1, 84, 1)

        prediction = model.predict(data_aux)
        predicted_character = labels_dict[int(np.argmax(prediction))]

        if last_prediction == predicted_character:
            repeat_count += 1
        else:
            last_prediction = predicted_character
            repeat_count = 1

        if repeat_count >= repeat_threshold:
            predicted_word.append(predicted_character)
            repeat_count = 0

        h, w, _ = frame.shape
        # Show the predicted character on the frame
        cv2.putText(frame, predicted_character, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        no_hand_frames += 1

    # If no hand is detected for the threshold duration, display the final predicted word
    if no_hand_frames >= no_hand_frames_threshold:
        if predicted_word:
            filtered_word = filter_repeated_characters(predicted_word)
            closest_match = find_closest_word(filtered_word, common_words)

            # Update the final word to be displayed
            final_word = closest_match

            # Reset the predicted word but keep displaying the final word
            predicted_word = []

    # Get the width of the frame to display text in the right corner
    frame_height, frame_width, _ = frame.shape

    # Get the width of the text to adjust it to the right corner
    text_size = cv2.getTextSize(final_word, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    text_x = frame_width - text_size[0] - 50  # Position the text to the right corner
    text_y = 200  # You can adjust the Y position if needed

    # Show the final predicted word on the frame in the right corner
    if final_word:
        cv2.putText(frame, final_word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
