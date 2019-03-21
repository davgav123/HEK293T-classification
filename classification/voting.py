from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

from os.path import join
import pandas as pd


def hard_voting_example(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
    knn = KNeighborsClassifier(n_neighbors=7)
    mnb = MultinomialNB()
    svc = SVC(C=100, kernel='poly', degree=1, gamma='scale')

    mdl = VotingClassifier(estimators=[('RFC', rfc), ('KNN', knn), ('MNB', mnb), ('SVM', svc)], voting='hard')
    mdl.fit(X_train, y_train)

    print('hard voting, random forest, knn, mnb and svc with poly kernel')
    print('Train set acc: {}'.format(mdl.score(X_train, y_train)))
    print('Test set acc: {}'.format(mdl.score(X_test, y_test)))


def hard_voting_with_weights_example(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
    knn = KNeighborsClassifier(n_neighbors=7)
    mnb = MultinomialNB()
    svc = SVC(C=100, kernel='poly', degree=1, gamma='scale')

    mdl = VotingClassifier(estimators=[('RFC', rfc), ('KNN', knn), ('MNB', mnb), ('SVM', svc)], voting='hard',
                           weights=[1.5, 1.7, 1, 2.5])
    mdl.fit(X_train, y_train)

    print('hard voting, with weights, random forest, knn, mnb and svc with poly kernel')
    print('Train set acc: {}'.format(mdl.score(X_train, y_train)))
    print('Test set acc: {}'.format(mdl.score(X_test, y_test)))


def hard_voting_svm_example(X_train, X_test, y_train, y_test):
    svc1 = SVC(C=100, kernel='poly', degree=1, gamma='scale')
    svc2 = SVC(C=100, kernel='poly', degree=3, gamma='scale')
    svc3 = SVC(C=100, kernel='poly', degree=1, gamma=0.1)
    svc4 = SVC(C=100, kernel='poly', degree=3, gamma=0.1)

    mdl = VotingClassifier(estimators=[('SVC1', svc1), ('SVC2', svc2), ('SVC3', svc3), ('SVC4', svc4)], voting='hard',
                           weights=[1.2, 1, 1.2, 1])
    mdl.fit(X_train, y_train)

    print('hard voting, with weights, svms with poly kernel')
    print('Train set acc: {}'.format(mdl.score(X_train, y_train)))
    print('Test set acc: {}'.format(mdl.score(X_test, y_test)))


def soft_voting_example(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
    dt = DecisionTreeClassifier()
    mnb = MultinomialNB()
    svc = SVC(C=100, kernel='poly', degree=1, gamma='scale', probability=True)

    mdl = VotingClassifier(estimators=[('RFC', rfc), ('DT', dt), ('MNB', mnb), ('SVM', svc)], voting='soft')
    mdl.fit(X_train, y_train)

    print('soft voting, method predict_proba for random patterns from training set')
    # random petterns from test set
    print('pattern1')
    print(mdl.predict_proba([X_test.iloc[4444]]))

    print('pattern2')
    print(mdl.predict_proba([X_test.iloc[800]]))

    print('pattern3')
    print(mdl.predict_proba([X_test.iloc[300]]))

    print('pattern4')
    print(mdl.predict_proba([X_test.iloc[2222]]))

    print('pattern5')
    print(mdl.predict_proba([X_test.iloc[1471]]))

    print('pattern6')
    print(mdl.predict_proba([X_test.iloc[0]]))


def main():
    df = pd.read_csv(join('..', 'data_preprocessed', 'combined_data.csv'))

    # target for our classification
    y = df['class']
    # rest of the data
    X = df.loc[:, df.columns != 'class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    hard_voting_example(X_train, X_test, y_train, y_test)
    hard_voting_with_weights_example(X_train, X_test, y_train, y_test)
    hard_voting_svm_example(X_train, X_test, y_train, y_test)
    soft_voting_example(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
