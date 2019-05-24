# -*- coding: utf-8 -*-
"""outlierClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1be8OgpLySwyvjjXbpaChMvcJzfc6sA8t
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

df = pd.read_csv('/content/gdrive/My Drive/ip_files/data/outliers.csv')
print(df.shape)

"""Split into data and classes"""

X = df.loc[:, df.columns != 'class']
y = df['class']

"""## Classification

Let's load some of the best models and classify outliers on them.
"""

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np

clf = load('/content/gdrive/My Drive/ip_files/models/svm_balanced_poly_C300_gamaScale_degree1.pkl')
predicted = clf.predict(X)

print('accuracy: {}'.format(np.round(accuracy_score(y, predicted), 4)))
print('recall score: \n{}'.format(np.round(recall_score(y, predicted, average=None), 4)))
print('precision score: \n{}'.format(np.round(precision_score(y, predicted, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y, predicted)))

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np

clf = load('/content/gdrive/My Drive/ip_files/models/svm_balanced_rbf_C300_gamaScale.pkl')
predicted = clf.predict(X)

print('accuracy: {}'.format(np.round(accuracy_score(y, predicted), 4)))
print('recall score: \n{}'.format(np.round(recall_score(y, predicted, average=None), 4)))
print('precision score: \n{}'.format(np.round(precision_score(y, predicted, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y, predicted)))

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np

clf = load('/content/gdrive/My Drive/ip_files/models/svm_sampled_class2_400_rbf_C300_gamaScale.pkl')
predicted = clf.predict(X)

print('accuracy: {}'.format(np.round(accuracy_score(y, predicted), 4)))
print('recall score: \n{}'.format(np.round(recall_score(y, predicted, average=None), 4)))
print('precision score: \n{}'.format(np.round(precision_score(y, predicted, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y, predicted)))

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np

clf = load('/content/gdrive/My Drive/ip_files/models/svm_sampled_class2_400_poly_C300_gamaScale_degree1s.pkl')
predicted = clf.predict(X)

print('accuracy: {}'.format(np.round(accuracy_score(y, predicted), 4)))
print('recall score: \n{}'.format(np.round(recall_score(y, predicted, average=None), 4)))
print('precision score: \n{}'.format(np.round(precision_score(y, predicted, average=None), 4)))
print('confusion matrix: \n{}'.format(confusion_matrix(y, predicted)))