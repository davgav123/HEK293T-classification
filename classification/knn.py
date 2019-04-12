from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from os.path import join
import gc


def knn(X_train, X_test, y_train, y_test):
    error = []
    ks = [3, 5, 7, 10, 15, 20, 50]

    # create models for different number of neighbors and calculate MAE
    for i in ks:
        knn_i = KNeighborsClassifier(n_neighbors=i)
        knn_i.fit(X_train, y_train)

        pred_i = knn_i.predict(X_test)
        error.append(np.abs(np.mean(pred_i != y_test)))

        gc.collect()

    # plot the graph of MAEs for every K
    plt.figure(figsize=(15, 8))
    plt.plot(ks, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)

    plt.xticks(ks)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Absolute Error')
    plt.show()


def knn_weighted(X_train, X_test, y_train, y_test):
    error_weighted = []
    ks = [3, 5, 7, 10, 15, 20, 50]

    # create models for different number of neighbors and calculate MAE
    for i in ks:
        knn_i = KNeighborsClassifier(n_neighbors=i, weights='distance')
        knn_i.fit(X_train, y_train)

        pred_i = knn_i.predict(X_test)
        error_weighted.append(np.abs(np.mean(pred_i != y_test)))

        gc.collect()

    # plot the graph of MAEs for every K
    plt.figure(figsize=(15, 8))
    plt.plot(ks, error_weighted, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)

    plt.xticks(ks)
    plt.title('Error Rate K Value with Weights')
    plt.xlabel('K Value')
    plt.ylabel('Mean Absolute Error')
    plt.show()


# we know that for seven neighbors we have the best results!
# so we will create KNN models (weighted and non-weighted) here
def the_best_k(X_train, X_test, y_train, y_test, k):
    print('KNN, k = 7, no weights:')

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print('Test set acc: {}'.format(knn.score(X_test, y_test)))
    print('MAE: {}'.format(np.abs(np.mean(y_pred != y_test))))  # mean absolute error

    print()
    print('confusion matrix: ')
    print(confusion_matrix(y_test, y_pred))
    print()
    print('classification report:')
    print(classification_report(y_test, y_pred))

    # create buffer between two models
    print()
    print()

    print('KNN, n = 7, weights = distance:')

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print('Test set acc: {}'.format(knn.score(X_test, y_test)))
    print('MAE: {}'.format(np.abs(np.mean(y_pred != y_test))))  # mean absolute error

    print()
    print('confusion matrix: ')
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
    # knn(X_train, X_test, y_train, y_test)
    # knn_weighted(X_train, X_test, y_train, y_test)

    the_best_k(X_train, X_test, y_train, y_test, k=7)


if __name__ == '__main__':
    main()
