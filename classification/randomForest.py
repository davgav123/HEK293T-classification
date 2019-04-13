from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

from os.path import join


def random_forest_clf(X_train, X_test, y_train, y_test, metrics='gini'):
    rfc = RandomForestClassifier(n_estimators=100, criterion=metrics)
    rfc.fit(X_train, y_train)

    print('criterion: {}'.format(metrics))
    print('Train set acc: {}'.format(rfc.score(X_train, y_train)))
    print('Test set acc: {}'.format(rfc.score(X_test, y_test)))

    y_pred = rfc.predict(X_test)
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    random_forest_clf(X_train, X_test, y_train, y_test, metrics='gini')
    random_forest_clf(X_train, X_test, y_train, y_test, metrics='entropy')


if __name__ == '__main__':
    main()
