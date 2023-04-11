import cv2
import os
import numpy as np
import tensorflow as tf

# Set paths to data and labels
data_path = '../data/video'
label_path = '../label/label.txt'

# Define hyperparameters
num_classes = 2
batch_size = 4
epochs = 5
learning_rate = 0.008

# Define a function to extract frames from a video and preprocess them
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

# Define a function to generate batches of data
def data_generator(data_path, label_path, max_frames=45):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if parts[1] == "No Spell" and parts[2] == "No Spell":
                continue
            label = parts[4]  # Load the ground truth from the 5th element
            if label == "survived if use spell properly":
                label = 1
            elif label == "dead even use spell properly":
                label = 0
            else:
                continue
            video_name = parts[0]  # Load the video name from the 1st element
            video_path = os.path.join(data_path, video_name)
            frames = extract_frames(video_path)
            if frames.shape[0] > max_frames:
                frames = frames[:max_frames]  # Truncate frames
            else:
                pad_width = ((0, max_frames - frames.shape[0]), (0, 0), (0, 0), (0, 0))
                frames = np.pad(frames, pad_width, 'constant')  # Pad frames
            yield frames, label

# Split data into training and validation sets
num_samples = len(os.listdir(data_path))
num_train_samples = int(num_samples * 0.8)
train_data = tf.data.Dataset.from_generator(lambda: data_generator(data_path, label_path, max_frames=45),
                                             output_types=(tf.float32, tf.int32),
                                             output_shapes=((45, 224, 224, 3), ()))
train_data = train_data.batch(batch_size).prefetch(1)

val_data = tf.data.Dataset.from_generator(lambda: data_generator(data_path, label_path, max_frames=45),
                                           output_types=(tf.float32, tf.int32),
                                           output_shapes=((45, 224, 224, 3), ()))
val_data = val_data.batch(batch_size).prefetch(1)

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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data,
          steps_per_epoch=num_train_samples // batch_size,
          validation_data=val_data,
          validation_steps=(num_samples - num_train_samples) // batch_size,
          epochs=epochs)

model.save('D:/pythonProject/model_save/if_use_spell_well.h5')

