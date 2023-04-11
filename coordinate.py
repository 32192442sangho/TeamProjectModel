import cv2

# Load the image
image = cv2.imread('./data/frame/13-1_KR-6331999909_01.webm_frame0.jpg')

# Create a window to display the image
cv2.namedWindow('image')
cv2.imshow('image', image)

# Define a mouse callback function to get the clicked point
def on_mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at point ({x}, {y})")

# Set the mouse callback function for the window
cv2.setMouseCallback('image', on_mouse_click)

# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()