import numpy
import cv2

from matplotlib import pyplot as plot

# read image
image = cv2.imread("images/test.jpg", cv2.IMREAD_COLOR)

# construct an ORB
# cv2.ORB() crashes python bc god is dead
orb = cv2.ORB_create()

# find the key points
keyPoints = orb.detect(image, None)
descriptors = orb.compute(image, keyPoints)

# display the final result of the ORB
imageWithKeyPoints = cv2.drawKeypoints(image, keyPoints, None, color=(0, 255, 0), flags=None)
plot.imshow(imageWithKeyPoints)
plot.show()
