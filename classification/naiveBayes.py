from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split

import pandas as pd
from os.path import join


def MNB(X_train, X_test, y_train, y_test):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    print('MultinomialNB train acc: {}'.format(mnb.score(X_train, y_train)))
    print('MultinomialNB test acc: {}'.format(mnb.score(X_test, y_test)))


def CNB(x_train, x_test, y_train, y_test):
    cnb = ComplementNB()
    cnb.fit(x_train, y_train)
    print('ComplementNB train acc: {}'.format(cnb.score(x_train, y_train)))
    print('ComplementNB test acc: {}'.format(cnb.score(x_test, y_test)))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    MNB(X_train. X_test, y_train, y_test)
    CNB(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
