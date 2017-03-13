import itertools
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
plt.style.use('seaborn-poster')

digits =load_digits()
print('we have %d samples' %len(digits.target))


fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
for i in range(64):
    ax = fig.add_subplot(8,8, i+1, xticks =[], yticks = [])
    ax.imshow(digits.images[i], cmap = plt.cm.binary, interpolation = 'nearest')
    ax.text(0, 7, str(digits.target[i]))


#split dataset to training (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=16)
print('Number of samples in training set: %d, number of samples in test set: %d'%(len(y_train), len(y_test)))

#pre-processing, transferring data to the range of 0 and 1
scaler = StandardScaler()
scaler.fit(X_train)

#apply the transformation to the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train the MLP classifier
#initialise ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='logistic', max_iter= 1000)

#train the classifier with training data
mlp.fit(X_train_scaled,y_train)


#activation function --> sigmoid (logistic)
#alpha value --> avoids overfitting
MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(30,30,30), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=None,
              shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
              verbose=False, warm_start=False)

#create function to print and plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title ='Confusion Matrix',
                          cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis =1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

 # predict results from the test data
    predicted = mlp.predict(X_test_scaled)

    # plot the confusion matrix
    cm = confusion_matrix(y_test, predicted)
    plot_confusion_matrix(cm, classes=digits.target_names,
                          title='Confusion matrix, without normalization')

    expected = y_test
    fig = plt.figure(figsize=(8, 8))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')

        # label the image with the target value
        if predicted[i] == expected[i]:
            ax.text(0, 7, str(predicted[i]), color='green')
        else:
            ax.text(0, 7, str(predicted[i]), color='red')