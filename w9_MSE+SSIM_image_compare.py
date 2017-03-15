import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import cv2


#Mean Squared Error algorithm finds the difference between two images
#images are the same size
def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

#comparison
img1 = cv2.imread("8x8.png")
img2 = cv2.imread("8x8v2.png")
mseval = mse(img1, img2)
ssimval = ssim(img1, img2,dynamic_range=255, gradient=True)

print(mseval)
print (ssimval)
