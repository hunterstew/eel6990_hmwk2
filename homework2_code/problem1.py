import cv2
import numpy as np
import matplotlib.pyplot as plt

# RGB
# src = cv2.imread('homework2_pics/prob1-1.jpg', cv2.IMREAD_UNCHANGED)
src = cv2.imread('homework2_pics/prob1-2.jpg', cv2.IMREAD_UNCHANGED)

RGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

#B image
B = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
B[:,:,0] = 0;
B[:,:,1] = 0;

#G image
G = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
G[:,:,0] = 0;
G[:,:,2] = 0;

#R image
R = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
R[:,:,1] = 0;
R[:,:,2] = 0;

rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
b = rgb[:,:,0]
g = rgb[:,:,1]
r = rgb[:,:,2]

#rgb
rgb[:,:,0] = b/(r+g+b+.1)
rgb[:,:,1] = g/(r+g+b+.1)
rgb[:,:,2] = r/(r+g+b+.1)

#r
srcr = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
srcr[:,:,0] = r/(r+g+b+.1)
srcr[:,:,1] = 0
srcr[:,:,2] = 0

#g
srcg = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
srcg[:,:,0] = 0
srcg[:,:,1] = g/(r+g+b+.1)
srcg[:,:,2] = 0

#b
srcb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
srcb[:,:,2] = b/(r+g+b+.1)
srcb[:,:,0] = 0
srcb[:,:,1] = 0

#LAB
lab = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)

#HSV
hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(hsv)

#display images
plt.subplot(341),plt.imshow(RGB),plt.title('RGB')
plt.xticks([]), plt.yticks([])
plt.subplot(342),plt.imshow(B),plt.title('B')
plt.xticks([]), plt.yticks([])
plt.subplot(343),plt.imshow(G),plt.title('G')
plt.xticks([]), plt.yticks([])
plt.subplot(344),plt.imshow(R),plt.title('R')
plt.xticks([]), plt.yticks([])
plt.subplot(345),plt.imshow(rgb),plt.title('rgb')
plt.xticks([]), plt.yticks([])
plt.subplot(346),plt.imshow(srcr),plt.title('r')
plt.xticks([]), plt.yticks([])
plt.subplot(347),plt.imshow(srcg),plt.title('g')
plt.xticks([]), plt.yticks([])
plt.subplot(348),plt.imshow(srcb),plt.title('b')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(231),plt.imshow(L),plt.title('L*')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(A),plt.title('a*')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(B),plt.title('b*')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(H),plt.title('H')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(S),plt.title('S')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(V),plt.title('V')
plt.xticks([]), plt.yticks([])
plt.show()