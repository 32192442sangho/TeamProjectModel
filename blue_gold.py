import cv2
import os
import pytesseract

# set path to the directory containing the videos
video_dir = r"C:\Users\AiA\Downloads\pythonProject\data\video"

# set the coordinates of the region of interest
left, top, width, height = (551, 7, 73, 26)

# set the path to the Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\AiA\Downloads\pythonProject\data\tesseract\tesseract.exe'

# loop over all files in the video directory
for filename in os.listdir(video_dir):
    # construct the full path to the video file
    video_path = os.path.join(video_dir, filename)

    # open video using cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)

    # read first frame
    ret, frame = cap.read()

    # extract region of interest
    roi = frame[top:top+height, left:left+width]

    # apply OCR on the region of interest to extract the string
    text = pytesseract.image_to_string(roi)

    # print the text detected for this video
    print(f"{filename}: {text.strip()}")

    # release the video capture object
    cap.release()

    # close any open windows
    cv2.destroyAllWindows()



