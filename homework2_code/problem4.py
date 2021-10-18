import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('homework2_pics/coins.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

plt.imshow(img)
plt.show()

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# from skimage import data
# from skimage.filters import threshold_multiotsu

# # Setting the font size for all plots.
# matplotlib.rcParams['font.size'] = 9

# # The input image.
# image = data.camera()

# # Applying multi-Otsu threshold for the default value, generating
# # three classes.
# thresholds = threshold_multiotsu(image)

# # Using the threshold values, we generate the three regions.
# regions = np.digitize(image, bins=thresholds)

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# # Plotting the original image.
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('Original')
# ax[0].axis('off')

# # Plotting the histogram and the two thresholds obtained from
# # multi-Otsu.
# ax[1].hist(image.ravel(), bins=255)
# ax[1].set_title('Histogram')
# for thresh in thresholds:
#     ax[1].axvline(thresh, color='r')

# # Plotting the Multi Otsu result.
# ax[2].imshow(regions, cmap='jet')
# ax[2].set_title('Multi-Otsu result')
# ax[2].axis('off')

# plt.subplots_adjust()

# plt.show()


from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2 as cv


img = cv.imread('homework2_pics/coins.png')  # data.coffee()

labels1 = segmentation.slic(img, compactness=20, n_segments=400, start_label=1)
out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 50)
out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                       figsize=(6, 8))

ax[0].imshow(out1.astype('uint8'))
ax[1].imshow(out2.astype('uint8'))

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
