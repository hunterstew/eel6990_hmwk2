import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('homework2_pics/room.jpg')          # queryImage
img2 = cv.imread('homework2_pics/room copy.jpg') # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv.imwrite('homework2_answers/problem3-sift.png', img3)
print(len(good)) # 302


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('homework2_pics/room.jpg')          # queryImage
img2 = cv.imread('homework2_pics/room copy.jpg')  # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv.drawMatches(img1, kp1, img2, kp2,
                      matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
cv.imwrite('homework2_answers/problem3-orb.png', img3)
print(len(matches))  # 154
