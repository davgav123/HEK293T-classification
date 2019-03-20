from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from os.path import join


def svm_rbf_kernel(X_train, X_test, y_train, y_test):
    clf = SVC(C=100, kernel='rbf', gamma=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('MAE: {}'.format(np.abs(np.mean(y_pred != y_test))))
    print(confusion_matrix(y_test, y_pred))


def svm_poly_kernel(X_train, X_test, y_train, y_test):
    error = []
    degrees = [1, 2, 3, 4, 5, 6]

    # gamma is 0.1 because of no particular reason
    for d in degrees:
        clf = SVC(C=100, kernel='poly', degree=d, gamma=0.1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        error.append(np.abs(np.mean(y_pred != y_test)))

    # plot results
    plt.figure(figsize=(15, 8))
    plt.plot(degrees, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)

    plt.xticks(degrees)
    plt.title('Error Rate for different degree values in SVM with polynomial kernel (gamma is 0.1)')
    plt.xlabel('Degree values')
    plt.ylabel('Mean Absolute Error')

    # now we will change gamma to 'scale'
    error = []
    degrees = [1, 2, 3, 4, 5, 6]

    for d in degrees:
        clf = SVC(C=100, kernel='poly', degree=d, gamma='scale')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        error.append(np.abs(np.mean(y_pred != y_test)))

    # and plot again
    plt.figure(figsize=(15, 8))
    plt.plot(degrees, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)

    plt.xticks(degrees)
    plt.title('Error Rate for different degree values in SVM with polynomial kernel (gamma = \'scale\')')
    plt.xlabel('Degree values')
    plt.ylabel('Mean Absolute Error')


# best results are for d = 1, kernel = 'poly'
def the_best_svm(X_train, X_test, y_train, y_test, d, c):
    clf = SVC(C=c, kernel='poly', degree=d, gamma='scale')
    clf.fit(X_train, y_train)

    print('Train set acc: {}'.format(clf.score(X_train, y_train)))
    print('Test set acc: {}'.format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)

    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()
    print('classification report:')
    print(classification_report(y_test, y_pred))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    svm_rbf_kernel(X_train, X_test, y_train, y_test)

    # now we test our best model, d = 1, kernel = 'poly', gamma = 'scale'
    # for different margin size
    the_best_svm(X_train, X_test, y_train, y_test, d=1, c=100)
    the_best_svm(X_train, X_test, y_train, y_test, d=1, c=300)


if __name__ == '__main__':
    main()
