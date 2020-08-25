################ Loading necessary library
import numpy as np
import cv2 as cv
import time
import argparse
from imutils.object_detection import non_max_suppression


################ Making argument parser and parsing argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="/home/hasan/Downloads/imagetext.png",help="path to input image")
ap.add_argument("-east", "--east", type=str, default="/home/hasan/Downloads/frozen_east_text_detection.pb", help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())


############# Loading and Changing Shape of the Image
# loading and taking only height and width of the image
image = cv.imread(args["image"])
original = image.copy()
(origH, origW) = image.shape[:2]

# New height and width of the image
(newWidth, newHeight) = (args['width'], args['height'])
ratioW = origW/float(newWidth)
ratioH = origH/float(newHeight)
# resizing of the new image
image = cv.resize(image, (newWidth, newHeight))
(H,W) = image.shape[:2]


#################### Defining two output layer of the EAST model
# First one for output probabilities
# Second one for to derive bounding box coordinates of text
layerNames = [
"feature_fusion/Conv_7/Sigmoid",
"feature_fusion/concat_3"]


################# Loading the EAST text-detector model
net = cv.dnn.readNet(args["east"])

################## Constructing a blob and perform forward pass
# making blob from the image
blob = cv.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

start_time = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end_time = time.time()
print("Total {:.6f} time to detect text".format(end_time-start_time))


################# Grabing number of rows and columns and initializing bounding box, confidence score
# Grabing number of rows and column from the scores
(numRows, numCols) = scores.shape[2:4]

rects = []
confidences = []

for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])



############## Applying non-max-suppression
# Applying non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scaling the bounding box coordinates based on the respective ratios
    startX = int(startX * ratioW)
    startY = int(startY * ratioH)
    endX = int(endX * ratioW)
    endY = int(endY * ratioH)
    # draw the bounding box on the image
    cv.rectangle(original, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv.imshow("Text Detection", original)
cv.waitKey(0)

