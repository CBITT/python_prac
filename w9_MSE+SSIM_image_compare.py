import ImageOps
import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import cv2
import ImageChops
import math, operator
import Image

#finds the difference between two images
#images are the same size


#Mean Squared Error (MSE) algorithm -> estimates the perceived change
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

#Structural Similarity Index (SSIM) algorithm -> models the perceived change
ssimval = ssim(img1, img2)

#Root mean squared error (RMSE) -> RMSE does not necessarily increase with
#                                  the variance of the errors. RMSE increases
#                                  with the variance of the frequency
#                                  distribution of error magnitudes.
rmse = math.sqrt(mse(img1,img2))

print ("MSE > ")
print(mseval)
print ("SSIM > ")
print (ssimval)
print ("RMSE > ")
print (rmse)

#ImageChops
#Returns the absolute value of the difference between the two images in pixels.
imgChops1 = Image.open("8x8.png")
imgChops2 = Image.open("8x8v2.png")
diff = ImageChops.difference(imgChops1,imgChops2)
print ("Pixel difference between two images > " )
print(diff)
