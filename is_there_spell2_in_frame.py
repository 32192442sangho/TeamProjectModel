import os
import cv2
import pyautogui
import numpy as np

# Define the region of interest on each frame
left = 167
top = 610
width = 30
height = 28

# Load the images from the specified directory
image_dir = r"C:\Users\AiA\Downloads\pythonProject\data\spell_images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
output_dir = r"C:\Users\AiA\Downloads\pythonProject\label\spell_2_for_each_frame"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Loop over all videos in the directory
video_dir = r"C:\Users\AiA\Downloads\pythonProject\data\video"
for video_file in os.listdir(video_dir):
    if not video_file.endswith(".webm"):
        continue  # Only process video files

    # Open the video file
    video_path = os.path.join(video_dir, video_file)
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video {video_path} with {total_frames} frames")

    # Create the output text file
    output_file = os.path.join(output_dir, os.path.splitext(video_file)[0] + ".txt")
    with open(output_file, "w") as f:
        f.write("0\n" * total_frames)  # Write 0 for each frame

    # Loop over all frames in the video
    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video.read()
        if not success:
            break  # End of video

        print(f"Processing frame {frame_count} with left={left}, top={top}, width={width}, height={height}")

        # Capture the region of interest in the frame
        region = frame[top:top + height, left:left + width]
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Loop over the images and search for each one in the region
        match_found = False
        for image in images:
            # Search for the image in the region using template matching
            result = cv2.matchTemplate(region_gray, image, cv2.TM_CCOEFF_NORMED)

            # Check if a match was found
            threshold = 0.44
            if cv2.minMaxLoc(result)[1] > threshold:
                print(f"Match found in frame {frame_count} of video {video_path}")

                # Update the output text file with 1 for the current frame
                with open(output_file, "r+") as f:
                    lines = f.readlines()
                    lines[frame_count] = "1\n"
                    f.seek(0)
                    f.writelines(lines)

                match_found = True
                # Return 1 or perform other actions as needed
                break

        if not match_found:
            # Update the output text file with 0 for the current frame
            with open(output_file, "r+") as f:
                lines = f.readlines()
                lines[frame_count] = "0\n"
                f.seek(0)
                f.writelines(lines)

        frame_count += 1

    # Release the video object
    video.release()