import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set the directory containing the images
DATA_DIR = './data'

# Loop through each directory and image
for dir_ in os.listdir(DATA_DIR):
    # Get all image files in the directory
    image_files = sorted([f for f in os.listdir(os.path.join(DATA_DIR, dir_)) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    # Process images from 300 to 305 (6 images in total)
    for img_file in image_files[299:305]:  # 299 is the index for the 300th image
        img_path = os.path.join(DATA_DIR, dir_, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the image
                mp_drawing.draw_landmarks(
                    img_rgb,  # Image to draw on
                    hand_landmarks,  # Hand landmarks detected
                    mp_hands.HAND_CONNECTIONS,  # Hand connections to draw
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # Style for landmarks
                    mp_drawing_styles.get_default_hand_connections_style()  # Style for connections
                )
        
        # Display the image with landmarks
        plt.figure()
        plt.imshow(img_rgb)
        plt.title(f"Image: {img_file}")  # Add title to show which image is being displayed

# Show the plots
plt.show()

# Close all figures
plt.close('all')