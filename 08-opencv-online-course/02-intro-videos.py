import cv2
import sys

# Print versions
print('=' * 50)
print(f"OpenCV Version: {cv2.__version__}")
print(f"Python {sys.version}")

# Create a video capture object and read in the input file to check to see that it can open the file
cap = cv2.VideoCapture('videos/vtest.avi')

while True:
    # Read
    ret, frame = cap.read()

    # Show frame
    cv2.imshow('frame', frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# # Open image
# img = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
