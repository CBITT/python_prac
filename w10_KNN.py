from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.figure

mnist = datasets.load_digits()

(train_data, test_data, train_labels, test_labels) = train_test_split(np.array(mnist.data),
                                        mnist.target, test_size=0.25, random_state=42)

(train_data,val_data, train_labels, val_labels) = train_test_split(train_data, train_labels,
                                test_size=0.1, random_state=84)

k_vals = range(1,30,2)
accuracies = []

for k in xrange(1,30,2):
    #train K-nearest neighbour classifier with the current value of k
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_data, train_labels)

    #evaluate the model and update the accuracies list
    score = model.score(val_data, val_labels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

#find the value of k that has the highest accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (k_vals[i],
	accuracies[i] * 100))

model = KNeighborsClassifier(n_neighbors=k_vals[i])
model.fit(train_data, train_labels)
predictions = model.predict(test_data)

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(test_labels, predictions))

# loop over a few random digits
for i in np.random.randint(0, high=len(test_labels), size=(5,)):
    # grab the image and classify it
    image = test_data[i]
    prediction = model.predict(image.reshape(1,-1) )[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels so we can see it better
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("Predicted digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Draw the figure.
fig = plt.figure(1)
plt.plot(k_vals, accuracies, 'ro', figure=fig)

fig.suptitle("Nearest Neighbor Classifier Accuracies")
fig.axes[0].set_xlabel("k (# of neighbors considered)")
fig.axes[0].set_ylabel("accuracy (% correct)");
fig.axes[0].axis([0, max(k_vals) + 1, 0, 1]);

plt.show()