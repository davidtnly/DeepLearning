import numpy as np
import cv2

# Create a video capture object and read in the input file to check to see that it can open the file
cap = cv2.VideoCapture('Videos/shore.mov')

# Read the synset classes file and strip off any characters
all_rows = open('synset_words.txt').read().strip().split('\n')

# Grab the different descriptions and not the id; we can use a list comprehension. 
# Find classes; looking for a space, and we don't want to include the id for r in all_rows; +1 for text after space
classes = [r[r.find(' ') + 1:] for r in all_rows]

# Read the caffe file
net = cv2.dnn.readNetFromCaffe('CaffeModel/bvlc_googlenet.prototxt', 'CaffeModel/bvlc_googlenet.caffemodel')

# Make sure we can open the video file
if cap.isOpened() == False:
    print('Cannot open file or video stream')

# Read and display video frames until either the video is completed or the user quits by typing an escape key
while True:

    # Call that return and frame and return true to see the frame
    ret, frame = cap.read()  # returns 2 parameters

    # ----------- blob snippet -------------------------------------------------
    # Create blob
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))  # change img to frame

    # Set Input
    net.setInput(blob)

    # Set forward pass to get the predictions for each of the 1,000 classes
    output = net.forward()
    # --------------------------------------------------------------------------

    r = 1
    for i in np.argsort(output[0])[::-1][:5]:
        # Text being printed on each frame
        txt = ' "%s" probability "%.3f" ' % (classes[i], output[0][i] * 100)  # (class, probability)

        # putText(frame, text, location of text, font, size, color, thickness)
        cv2.putText(frame, txt, (0, 25 + 40 * r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        r += 1

    # If true, show frame
    if ret == True:
        cv2.imshow('Frame', frame)

        # Pass a number greater than zero for videos; OpenCV suggests 25 milliseconds
        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break

# Once we're done, release the video and remove windows        
cap.release()
cv2.destroyAllWindows()