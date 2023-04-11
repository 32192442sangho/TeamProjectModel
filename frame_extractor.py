import os
import cv2

# Specify the directory where the videos are stored
video_dir = r"./data/video"

# Specify the directory where the frames will be saved
frame_dir = r"./data/frame"

# Get a list of all video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".webm")]

# Loop over the first 100 video files in the list
for i, filename in enumerate(video_files[:100]):
    # Load the video
    video_path = os.path.join(video_dir, filename)
    video = cv2.VideoCapture(video_path)

    # Loop over all frames in the video
    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video.read()
        if not success:
            break  # End of video

        # Save the frame as an image file
        frame_path = os.path.join(frame_dir, f"{filename}_frame{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video object
    video.release()

    # Print progress
    print(f"Processed video {i+1} of {min(100, len(video_files))}: {filename}")

    # Check if we have processed the first 100 videos
    if i == 99:
        break