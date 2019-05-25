# -*- coding: utf-8 -*-
"""balancingClasses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qaSGI-6X7p9aRuFB-cN6WyJyjH531Nlt

# Balancing classes

Classes in our dataset are not balanced. Here, we will try to address that problem.
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

df = pd.read_csv('/content/gdrive/My Drive/ip_files/data/data_without_outliers.csv')
print(df.shape)

"""## Naive

First we will try naive approach. Out of 17000ish rows, only 66 are of the class2, so we will delete them.
"""

from sklearn.model_selection import train_test_split

df = df.loc[df['class'] != 'class2']
print('new shape: {}'.format(df.shape))

y = df['class']
X = df.loc[:, df.columns != 'class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# free some of the much needed memory
del X
del y
del df

import gc
gc.collect()

"""Now we will build some of the best models from before.

### KNN
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('accuracy train: {}'.format(np.round(knn.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""Accuracy on test set is higher, but this classifier struggles with class3.

### Decision trees
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='gini', class_weight='balanced')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""These aren't much better. Gini without class weights is the best model out of all decision trees.

### SVM
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='rbf', gamma='scale', class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='poly', gamma='scale', degree=1, class_weight=None)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='poly', gamma='scale', degree=1, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""### Neural networks"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

n = len(X_train.loc[0, :])
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(n // 16, n // 64, n // 128))
clf.fit(X_train, y_train)
                    
y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""### Voting"""

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

svc1 = SVC(C=100, kernel='poly', degree=1, gamma='scale')
svc2 = SVC(C=100, kernel='rbf', degree=1, gamma='scale', class_weight='balanced')
svc3 = SVC(C=300, kernel='rbf', gamma='scale', class_weight='balanced')
svc4 = SVC(C=300, kernel='rbf', gamma='scale')

vclf = VotingClassifier(estimators=[('SVC1', svc1), ('SVC2', svc2), ('SVC3', svc3), ('SVC4', svc4)], voting='hard')
vclf.fit(X_train, y_train)

y_pred = vclf.predict(X_test)

print('accuracy train: {}'.format(np.round(vclf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""## Sampling

Here we will, using sklearn.utils.resample add some of the patterns of class2 into our data, and build models on that.
"""

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import gc

X = df.loc[:, df.columns != 'class']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

del df
del y
gc.collect()

X = pd.concat([X_train, y_train], axis=1)

gc.collect()

class2 = X.loc[X['class'] == 'class2']
rest_of_the_data = X.loc[X['class'] != 'class2']

del X
gc.collect()

# class2 = X[X.class == 'class2']
# rest_of_the_data = X[X.class != 'class2']

class2_upsampled = resample(class2, random_state=27, n_samples=400, replace=True)
upsampled = pd.concat([rest_of_the_data, class2_upsampled])

gc.collect()

from collections import Counter

classes = upsampled['class']

print('classes count:')
print(sorted(Counter(classes).items()))

X_train = upsampled.loc[:, upsampled.columns != 'class']
y_train = upsampled['class']

del upsampled

"""### SVM"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from joblib import dump
dump(clf, '/content/gdrive/My Drive/ip_files/models/svm_sampled_class2_400_rbf_C300_gamaScale.pkl')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='poly', gamma='scale', degree=1, class_weight=None)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from joblib import dump
dump(clf, '/content/gdrive/My Drive/ip_files/models/svm_sampled_class2_400_poly_C300_gamaScale_degree1.pkl')

"""### Voting"""

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

svc1 = SVC(C=100, kernel='poly', degree=1, gamma='scale')
svc2 = SVC(C=300, kernel='poly', degree=1, gamma='scale')
svc3 = SVC(C=100, kernel='rbf', gamma='scale')
svc4 = SVC(C=300, kernel='rbf', gamma='scale')

vclf = VotingClassifier(estimators=[('SVC1', svc1), ('SVC2', svc2), ('SVC3', svc3), ('SVC4', svc4)], voting='hard')
vclf.fit(X_train, y_train)

y_pred = vclf.predict(X_test)

print('accuracy train: {}'.format(np.round(vclf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))


from joblib import dump
dump(vclf, '/content/gdrive/My Drive/ip_files/models/voting_sampled_class2_400_svm_examples.pkl')
print()

from joblib import dump
dump(vclf, '/content/gdrive/My Drive/ip_files/models/voting_sampled_class2_400_svm_examples.pkl')

"""## Balance

Here we will change our data in that way that every class has same number of patterns. We will choose 2500. That number is selected because data frame with 17500 patterns will fit our RAM memory.
"""

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import gc

X = df.loc[:, df.columns != 'class']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

del df
del y
gc.collect()

X = pd.concat([X_train, y_train], axis=1)

gc.collect()

class1 = X.loc[X['class'] == 'class1']
class2 = X.loc[X['class'] == 'class2']
class3 = X.loc[X['class'] == 'class3']
class4 = X.loc[X['class'] == 'class4']
class5 = X.loc[X['class'] == 'class5']
class6 = X.loc[X['class'] == 'class6']
class7 = X.loc[X['class'] == 'class7']

del X
gc.collect()

class1_sampled = resample(class1, random_state=27, n_samples=2500, replace=True)
del class1
class2_sampled = resample(class2, random_state=27, n_samples=2500, replace=True)
del class2
class3_sampled = resample(class3, random_state=27, n_samples=2500, replace=True)
del class3
class4_sampled = resample(class4, random_state=27, n_samples=2500, replace=True)
del class4
class5_sampled = resample(class5, random_state=27, n_samples=2500, replace=True)
del class5
class6_sampled = resample(class6, random_state=27, n_samples=2500, replace=True)
del class6
class7_sampled = resample(class7, random_state=27, n_samples=2500, replace=True)
del class7

gc.collect()

balanced = pd.concat([class1_sampled, class2_sampled,
                       class3_sampled, class4_sampled,
                       class5_sampled, class6_sampled,
                       class7_sampled])

del class1_sampled
del class2_sampled
del class3_sampled
del class4_sampled
del class5_sampled
del class6_sampled
del class7_sampled

gc.collect()

from collections import Counter

classes = balanced['class']

print('classes count:')
print(sorted(Counter(classes).items()))

"""### Decision trees"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np


dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print('accuracy train: {}'.format(np.round(dtc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

"""### SVM"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from joblib import dump
dump(clf, '/content/gdrive/My Drive/ip_files/models/svm_balanced_rbf_C300_gamaScale.pkl')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

clf = SVC(C=300, kernel='poly', gamma='scale', degree=1, class_weight=None)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('accuracy train: {}'.format(np.round(clf.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))

from joblib import dump
dump(clf, '/content/gdrive/My Drive/ip_files/models/svm_balanced_poly_C300_gamaScale_degree1.pkl')

"""### Gradient boosting"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
import numpy as np

gbc = GradientBoostingClassifier(loss='deviance', n_estimators=500)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)

print('accuracy train: {}'.format(np.round(gbc.score(X_train, y_train), 4)))
print('accuracy test: {}'.format(np.round(accuracy_score(y_test, y_pred), 4)))
print('recall score: {}'.format(np.round(recall_score(y_test, y_pred, average=None), 4)))
print('f1_score: {}'.format(np.round(f1_score(y_test, y_pred, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))