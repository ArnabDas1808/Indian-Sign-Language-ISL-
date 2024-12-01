import os

# Create the main "data" folder
main_folder = "data"
os.makedirs(main_folder, exist_ok=True)

# Create subfolders 0 to 35
for i in range(36):
    subfolder = os.path.join(main_folder, str(i))
    os.makedirs(subfolder, exist_ok=True)

print("Folders created successfully.")