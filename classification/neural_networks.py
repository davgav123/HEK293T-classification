from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from os.path import join
import pandas as pd


def n_network_classification_solver(X_train, X_test, y_train, y_test, sol='adam'):
    n = len(X_train.loc[0, :])
    clf = MLPClassifier(solver=sol, hidden_layer_sizes=(n // 16, n // 64))
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

    print('reading finished')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # n_network_classification_solver(X_train, X_test, y_train, y_test, 'lbfgs')
    n_network_classification_solver(X_train, X_test, y_train, y_test, 'adam')
    # n_network_classification_solver(X_train, X_test, y_train, y_test, 'sgd')


if __name__ == '__main__':
    main()
