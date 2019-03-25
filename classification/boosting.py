from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from os.path import join
import pandas as pd


def boost_svm(X_train, X_test, y_train, y_test):
    svm = SVC(C=100, kernel='poly', degree=1, gamma='scale', probability=True)

    abclf = AdaBoostClassifier(base_estimator=svm, n_estimators=3)
    abclf.fit(X_train, y_train)

    # print('Train set acc: {}'.format(abclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(abclf.score(X_test, y_test)))


def boost_naive_bayes(X_train, X_test, y_train, y_test):
    mnb = MultinomialNB()

    abclf = AdaBoostClassifier(base_estimator=mnb, n_estimators=100)
    abclf.fit(X_train, y_train)

    print('MNB Train set acc: {}'.format(abclf.score(X_train, y_train)))
    print('MNB Test set acc: {}'.format(abclf.score(X_test, y_test)))

    print('Complement Naive Bayes for different number of estimators')
    cnb = ComplementNB()

    for i in [50, 100, 200]:
        abclf = AdaBoostClassifier(base_estimator=cnb, n_estimators=i)
        abclf.fit(X_train, y_train)

        print('n_estimators={}'.format(i))
        print('CNB Train set acc: {}'.format(abclf.score(X_train, y_train)))
        print('CNB Test set acc: {}'.format(abclf.score(X_test, y_test)))


def boost_decision_tree(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier()

    abclf = AdaBoostClassifier(base_estimator=dtc, n_estimators=50)
    abclf.fit(X_train, y_train)

    print('Decision trees')
    print('Train set acc: {}'.format(abclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(abclf.score(X_test, y_test)))


def boost_random_forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=100)

    abclf = AdaBoostClassifier(base_estimator=rfc, n_estimators=250)
    abclf.fit(X_train, y_train)

    print('Random forest')
    print('Train set acc: {}'.format(abclf.score(X_train, y_train)))
    print('Test set acc: {}'.format(abclf.score(X_test, y_test)))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # boost_naive_bayes(X_train, X_test, y_train, y_test)
    # boost_decision_tree(X_train, X_test, y_train, y_test)
    # boost_random_forest(X_train, X_test, y_train, y_test)
    boost_svm(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
