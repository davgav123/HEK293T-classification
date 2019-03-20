from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import numpy as np
import gc


def scaled_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test):
    mnb = MultinomialNB()
    mnb.fit(X_train_scaled, y_train)
    print('MultinomialNB train acc: {}'.format(mnb.score(X_train_scaled, y_train)))
    print('MultinomialNB test acc: {}'.format(mnb.score(X_test_scaled, y_test)))

    cnb = ComplementNB()
    cnb.fit(X_train_scaled, y_train)
    print('ComplementNB train acc: {}'.format(cnb.score(X_train_scaled, y_train)))
    print('ComplementNB test acc: {}'.format(cnb.score(X_test_scaled, y_test)))


def scaled_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    error = []
    ks = [5, 7, 15, 25, 50, 75, 100, 125]

    for i in ks:
        knn_i = KNeighborsClassifier(n_neighbors=i)
        knn_i.fit(X_train_scaled, y_train)

        pred_i = knn_i.predict(X_test_scaled)
        error.append(np.abs(np.mean(pred_i != y_test)))

        gc.collect()

    # plot the results
    plt.figure(figsize=(15, 8))
    plt.plot(ks, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)

    plt.xticks(ks)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Absolute Error')


def scaled_svm_rbf(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import numpy as np

    clf = SVC(C=100, kernel='rbf', gamma=0.1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    print('MAE: {}'.format(np.abs(np.mean(y_pred != y_test))))
    print(confusion_matrix(y_test, y_pred))


def scaled_svm_poly(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import numpy as np

    clf = SVC(C=100, kernel='poly', degree=3, gamma=0.1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    print('MAE: {}'.format(np.abs(np.mean(y_pred != y_test))))
    print(confusion_matrix(y_test, y_pred))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    # scale data
    scl = MinMaxScaler()
    X = scl.fit_transform(X)

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaled_naive_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
    scaled_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    scaled_svm_rbf(X_train_scaled, X_test_scaled, y_train, y_test)
    scaled_svm_poly(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == '__main__':
    main()
