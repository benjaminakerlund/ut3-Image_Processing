import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('QtAgg')
print(matplotlib.matplotlib_fname())

''' Code for matplotlib 
import matplotlib.image as mpimg


# Load the image
img = mpimg.imread('../cargo.jpg')

# Show the image
imgplot = plt.imshow(img)
plt.show()


# testing with matplotlib
img = cv2.cvtColor(image_cargo, cv2.COLOR_BGR2RGB)


# Create subplots
plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')

plt.show() '''


''' Code for cv2 library '''
import cv2

# Load and display image
image_cargo = cv2.imread('../cargo.jpg')
image_cargo_gray = cv2.cvtColor(image_cargo, cv2.COLOR_RGB2GRAY)

cv2.imshow('image1', image_cargo)
cv2.imshow('image2', image_cargo_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

a = np.array([[1], [1]])
np.tile(a, [1,1,1])

print("This is a:")
print(a)

