import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("C:/Users/AiA/Downloads/pythonProject/model_save/spell1_use_epoch1000.h5")

# Load the spell images
spell_images = []
for i in range(1, 17):
    img = cv2.imread(f"C:/Users/AiA/Downloads/pythonProject/data/spell_images/{i}.jpg", cv2.IMREAD_GRAYSCALE)
    spell_images.append(img)

# Load the video
cap = cv2.VideoCapture("C:/Users/AiA/Downloads/pythonProject/test/13-1_KR-6336134778_49.webm")

# Iterate over each frame and classify
results = []
while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Extract the screen region
    screen = frame[610:610 + 28, 145:145 + 26]

    # Convert to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # Compare to each spell image
    match = False
    for spell_img in spell_images:
        # Resize the spell image to match the screen size
        spell_img_resized = cv2.resize(spell_img, (26, 28))

        # Compare the images
        if np.array_equal(screen_gray, spell_img_resized):
            results.append(1)
            match = True
            break

    if not match:
        results.append(0)
print(len(results))
# Convert the results to a numpy array
X_test = np.array(results).reshape(1, -1)

# Classify the frames
y_pred = model.predict(X_test)

# Print the classification result
if y_pred[0] == 1:
    print("The spell not used.")
else:
    print("The spell used.")

# Release the video capture
cap.release()