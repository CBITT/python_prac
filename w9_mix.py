from sklearn import datasets, svm, metrics
from PIL import Image


digits = datasets.load_digits()

#print(digits)


def normalizeImage():
    image = Image.open("number.png")
    image = image.convert('LA')     # "LA" (grayscale + alpha).
    image.thumbnail((8,8),Image.ANTIALIAS)
    image.save("numberOut.png")

