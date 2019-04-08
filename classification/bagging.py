from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
from os.path import join


def knn_bagging(X_train, X_test, y_train, y_test, k=7, n_est=50, max_sampl=1):
    knn = KNeighborsClassifier(n_neighbors=k)

    bclf = BaggingClassifier(base_estimator=knn, n_estimators=n_est, max_samples=max_sampl)
    bclf.fit(X_train, y_train)

    print('Train set acc: {}'.format(bclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(bclf.score(X_test, y_test)))


def svm_bagging(X_train, X_test, y_train, y_test, n_est=50, max_sampl=1):
    svm = SVC(C=100, kernel='poly', degree=1, gamma='scale')

    bclf = BaggingClassifier(base_estimator=svm, n_estimators=n_est, max_samples=max_sampl)
    bclf.fit(X_train, y_train)

    print('Train set acc: {}'.format(bclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(bclf.score(X_test, y_test)))


def neural_networks_bagging(X_train, X_test, y_train, y_test, n_est=50):
    n = len(X_train.loc[0, :])
    nn = MLPClassifier(solver='adam', hidden_layer_sizes=(n // 16, n // 64))

    clf = BaggingClassifier(base_estimator=nn, n_estimators=n_est)
    clf.fit(X_train, y_train)

    print('Test set acc: {}'.format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # params n_est and max_sampl should have greater value
    # knn_bagging(X_train, X_test, y_train, y_test, 7, 1, 0.1)
    # svm_bagging(X_train, X_test, y_train, y_test, 20, 0.3)
    neural_networks_bagging(X_train, X_test, y_train, y_test, n_est=6)


if __name__ == '__main__':
    main()
