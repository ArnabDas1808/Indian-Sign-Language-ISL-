import os
import shutil
import cv2

def copy_images_with_sequential_numbering(src_root_dir, dest_root_dir, folder_pairs):
    for src_folder, dest_folder in folder_pairs:
        src_folder_path = os.path.join(src_root_dir, src_folder)
        dest_folder_path = os.path.join(dest_root_dir, dest_folder)
        
        # Check if the source folder exists
        if not os.path.isdir(src_folder_path):
            print(f"Source folder not found: {src_folder_path}")
            continue
        
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder_path, exist_ok=True)
        
        print(f"Copying from folder {src_folder_path} to folder {dest_folder_path}")
        
        # Find the highest numbered image in the destination folder
        existing_images = [f for f in os.listdir(dest_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        start_number = 0
        if existing_images:
            highest_number = max([int(os.path.splitext(f)[0]) for f in existing_images])
            start_number = highest_number + 1
        
        # Process each image in the source folder
        for filename in os.listdir(src_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                src_file_path = os.path.join(src_folder_path, filename)
                
                # Generate new filename with sequential number
                new_filename = f"{start_number:03d}{os.path.splitext(filename)[1]}"
                dest_file_path = os.path.join(dest_folder_path, new_filename)
                
                # Copy the image
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} -> {dest_file_path}")
                
                start_number += 1

# Specify the root directories
src_root_directory = r"E:\SIH - Indian Sign Language\data"
dest_root_directory = r"E:\Indian Sign Language\data"

# Specify the source and destination folder pairs
folder_pairs = [
    ('24', '5'),
]

# Call the function to copy images
copy_images_with_sequential_numbering(src_root_directory, dest_root_directory, folder_pairs)

print("All specified folder pairs have been processed.")