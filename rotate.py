import os
import cv2

def flip_images_in_directory(root_dir, folder_to_process, start_num, end_num):
    folder_path = os.path.join(root_dir, folder_to_process)
    
    # Check if the folder exists
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_to_process}")
        
        # Process images from start_num to end_num
        for i in range(start_num, end_num + 1):
            filename = f"{i:03d}.jpg"  # Assuming the images are named as 000.jpg, 001.jpg, etc.
            file_path = os.path.join(folder_path, filename)
            
            if os.path.exists(file_path):
                # Read the image
                img = cv2.imread(file_path)
                
                if img is not None:
                    # Flip the image horizontally (left-right flip)
                    flipped_img = cv2.flip(img, 1)
                    
                    # Save the flipped image, overwriting the original
                    cv2.imwrite(file_path, flipped_img)
                    print(f"Flipped and saved: {file_path}")
                else:
                    print(f"Could not read image: {file_path}")
            else:
                print(f"Image not found: {file_path}")
    else:
        print(f"Folder not found: {folder_path}")

# Specify the root directory
root_directory = r"E:\Indian Sign Language\data"

# Specify the folder you want to process
folder_to_process = '3'

# Specify the range of images to process
start_number = 0
end_number = 1799

# Call the function to flip images
flip_images_in_directory(root_directory, folder_to_process, start_number, end_number)

print("All specified images have been processed.")