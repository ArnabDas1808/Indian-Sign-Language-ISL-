import pickle
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import Counter
import os

# Function to count images in the data directory
def count_images(data_dir):
    total_images = 0
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            total_images += len(images)
    return total_images

# Specify the path to your data directory
data_dir = r"E:\Indian Sign Language\data"

# Count the total number of images
total_images = count_images(data_dir)
print(f"Total number of images in directory: {total_images}")

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

print(f"Number of entries in data: {len(data)}")
print(f"Number of entries in labels: {len(labels)}")

# Use the minimum of total_images and len(data) to avoid index out of range errors
num_to_process = min(total_images, len(data))

processed_data = []
processed_labels = []
#print(num_to_process)

for i in range(num_to_process):
    if len(data[i]) == 42:
        data_one = data[i][:]
        #data_two = []
        for j in range(0, 42):
            #data_two.append(data_one[j])
            #data_two.append(data_one[j])
            data_one.append(0)
        processed_data.append(data_one)
        processed_labels.append(labels[i])
    elif len(data[i]) == 84:
        processed_data.append(data[i])
        processed_labels.append(labels[i])
    else:
        print(f"Skipping data point {i} with unexpected length: {len(data[i])}")

print(f"Number of processed data points: {len(processed_data)}")


# Convert to NumPy arrays
data = np.asarray(processed_data)
labels = np.asarray(processed_labels)

# Reshape data for CNN
data = data.reshape((data.shape[0], data.shape[1], 1))

# Check for any classes with only one sample and remove them
label_counts = Counter(labels)
labels_to_remove = [label for label, count in label_counts.items() if count < 2]

if labels_to_remove:
    indices_to_keep = [i for i in range(len(labels)) if labels[i] not in labels_to_remove]
    data = data[indices_to_keep]
    labels = labels[indices_to_keep]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels if len(labels_to_remove) == 0 else None)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Build CNN model
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test, y_test)
print(f'CNN Test Accuracy: {cnn_accuracy * 100:.2f}%')

# Save the CNN model
cnn_model.save('cnn_model.h5')