import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # Import TensorFlow

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
expected_num_features = 84  # Change this to match the input shape of your model

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    predicted_character = ''  # Initialize the predicted character as an empty string

    if results.multi_hand_landmarks:  # Ensure hands are detected
        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Collect all (x, y) coordinates for both hands
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                all_landmarks.append((x, y))

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        if len(results.multi_hand_landmarks) > 1:
            # Calculate the bounding box coordinates for both hands together
            x_min = min(lm[0] for lm in all_landmarks)
            y_min = min(lm[1] for lm in all_landmarks)
            x_max = max(lm[0] for lm in all_landmarks)
            y_max = max(lm[1] for lm in all_landmarks)

            # Convert normalized coordinates to image coordinates
            h, w, _ = frame.shape
            x_min = int(x_min * w) - 10
            y_min = int(y_min * h) - 10
            x_max = int(x_max * w) + 10
            y_max = int(y_max * h) + 10

            # Ensure the coordinates are within the image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Prepare the data_aux for prediction by ensuring it matches expected_num_features
            data_aux = [coord for landmark in all_landmarks for coord in landmark]
            # Trim or pad data_aux to match the expected number of features
            data_aux = data_aux[:expected_num_features] + [0.0] * (expected_num_features - len(data_aux))

            # Reshape data_aux to match model's expected input shape
            data_aux = np.array(data_aux).reshape(1, 84, 1)

            # Make a prediction using the model
            prediction = model.predict(data_aux)
            predicted_character = labels_dict[int(np.argmax(prediction))]

            # Draw the rectangle and predicted character
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            # Only process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            x_ = []
            y_ = []
            H, W, _ = frame.shape

            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Collect the (x, y) coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Ensure the shape of data_aux matches the model's input shape
            data_aux = data_aux[:expected_num_features] + [0.0] * (expected_num_features - len(data_aux))

            # Reshape data_aux to match model's expected input shape
            data_aux = np.array(data_aux).reshape(1, 84, 1)

            prediction = model.predict(data_aux)
            predicted_character = labels_dict[int(np.argmax(prediction))]

            # Draw the rectangle and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
