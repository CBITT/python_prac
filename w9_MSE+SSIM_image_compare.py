import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import cv2


#finds the difference between two images
#Mean Squared Error algorithm -> estimates the perceived change
#Structural Similarity Index algorithm -> models the perceived change
#images are the same size
def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

img1 = cv2.imread("8x8.png")
img2 = cv2.imread("8x8v2.png")

#grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

mseval = mse(img1, img2)
ssimval = ssim(img1, img2)

print(mseval)
print (ssimval)
