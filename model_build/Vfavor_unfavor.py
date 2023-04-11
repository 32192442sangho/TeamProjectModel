import cv2
import tensorflow as tf
import numpy as np
import os

# Set paths to data and labels
data_path = '../data/video'
label_path = '../label/label.txt'

# Define hyperparameters
num_classes = 3
batch_size = 16
epochs = 7
learning_rate = 0.001


# Define a function to extract frames from videos
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if num_frames % int(fps/3) == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = tf.keras.preprocessing.image.img_to_array(frame)
                frames.append(frame)
            num_frames += 1
        else:
            break
    cap.release()
    print(f"Number of frames for {video_path}: {len(frames)}")
    return np.array(frames)


# Load data and labels
def load_data(data_path, label_path, max_frames=45):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = []
        video_names = []
        for line in lines:
            print(line)
            parts = line.split(',')
            print(parts)
            label = parts[3]  # Load the ground truth from the 5th element
            print(label)
            if label == "favorable":
                labels.append(1)
            elif label == "unfavorable":
                labels.append(0)
            elif label == "neck and neck":
                labels.append(2)
            else:
                continue

            video_name = parts[0]  # Load the video name from the 1st element
            print(video_name)
            video_path = os.path.join(data_path, video_name)
            video_names.append(video_path)
    data = []
    print(labels)
    print(video_names)
    print(data)
    for video_path in video_names:
        frames = extract_frames(video_path)
        if frames.shape[0] > max_frames:
            frames = frames[:max_frames]  # Truncate frames
        else:
            pad_width = ((0, max_frames - frames.shape[0]), (0, 0), (0, 0), (0, 0))
            frames = np.pad(frames, pad_width, 'constant')  # Pad frames
        frames_tensor = tf.convert_to_tensor(frames)  # Convert frames to tensor
        data.append(frames_tensor)
    return tf.stack(data), np.array(labels)  # Stack the tensors to create a batch


x_train, y_train = load_data(data_path, label_path)

# Split data into training and validation sets
num_samples = len(x_train)
num_train_samples = int(num_samples * 0.8)
train_indices = tf.constant(np.random.choice(num_samples, num_train_samples, replace=False))
val_indices = tf.constant(np.setdiff1d(np.arange(num_samples), train_indices))
x_val, y_val = tf.gather(x_train, val_indices), tf.gather(y_train, val_indices)
x_train, y_train = tf.gather(x_train, train_indices), tf.gather(y_train, train_indices)

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=(45, 224, 224, 3)),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=epochs)

# Save model
model.save('D:/pythonProject/model_save/favor_unfavor.h5')