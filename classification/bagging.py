from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas as pd
from os.path import join


def knn_bagging(X_train, X_test, y_train, y_test, k=7, n_est=1, max_sampl=0.1):
    knn = KNeighborsClassifier(n_neighbors=k)

    bclf = BaggingClassifier(base_estimator=knn, n_estimators=n_est, max_samples=max_sampl)
    bclf.fit(X_train, y_train)

    print('Train set acc: {}'.format(bclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(bclf.score(X_test, y_test)))


def svm_bagging(X_train, X_test, y_train, y_test, n_est=1, max_sampl=0.1):
    svm = SVC(C=100, kernel='poly', degree=1, gamma='scale')

    bclf = BaggingClassifier(base_estimator=svm, n_estimators=n_est, max_samples=max_sampl)
    bclf.fit(X_train, y_train)

    print('Train set acc: {}'.format(bclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(bclf.score(X_test, y_test)))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # params n_est and max_sampl should have greater value
    knn_bagging(X_train, X_test, y_train, y_test, 7, 1, 0.1)
    svm_bagging(X_train, X_test, y_train, y_test, 20, 0.3)


if __name__ == '__main__':
    main()
