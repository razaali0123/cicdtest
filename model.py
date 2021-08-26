import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle

df = pd.read_csv("iris.data")

le = preprocessing.LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
df = df.sample(frac = 1)
xtr = df.iloc[:, :-1]
ytr = df.iloc[:, -1]
clf = SVC()
clf.fit(xtr, ytr)
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# if __name__ == "__main__":
