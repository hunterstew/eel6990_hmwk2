import cv2
import numpy as np
import matplotlib.pyplot as plt


# #original image
src = cv2.imread('homework2_pics/prob2.jpg', cv2.IMREAD_UNCHANGED)
# print(src.shape)
src =cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
user_red = float(input("Enter red value between 0 and 1:"))
user_green = float(input("Enter green value between 0 and 1:"))
user_blue = float(input("Enter blue value between 0 and 1:"))
src[:,:,0] = src[:,:,0] * user_red
src[:,:,1] = src[:,:,1] * user_green
src[:,:,2] = src[:,:,2] * user_blue

plt.imshow(src)
plt.show()
cv2.imwrite('homework2_answers/problem2.png', src)