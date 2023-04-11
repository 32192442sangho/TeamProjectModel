import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

epoch = 25
# Set directory path for data and ground truth files
data_dir = '../label/spell_2_for_each_frame'
truth_file = '../label/spell_2_use_truth.txt'

# Read data and ground truth
data = []
for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), 'r') as f:
        file_data = [int(line) for line in f.readlines()]
        if len(file_data) < 940:
            file_data += [3] * (940 - len(file_data))
        data.append(file_data)
data = np.array(data)

truth = np.genfromtxt(truth_file)

# Convert labels to categorical format
truth_cat = to_categorical(truth, num_classes=3)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, truth_cat, test_size=0.2, random_state=42)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train.argmax(axis=1))

# Make predictions on test set
y_pred_lr = lr_model.predict(X_test)

# Calculate accuracy
accuracy_lr = accuracy_score(y_test.argmax(axis=1), y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Train neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(940,)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epoch)

# Make predictions on test set
y_pred_nn = model.predict(X_test)
y_pred_nn = np.argmax(y_pred_nn, axis=1)

# Calculate accuracy
accuracy_nn = accuracy_score(y_test.argmax(axis=1), y_pred_nn)
print("Neural Network Accuracy:", accuracy_nn)

# Save model as h5 file
model.save(f'D:/pythonProject/model_save/spell2_use_epoch{epoch}.h5')