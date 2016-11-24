import csv
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np

def test_svm(ft, lbl):
    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.2, random_state=0)
    clf_svm = svm.SVC()
    gs = GridSearchCV(clf_svm, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gs.fit(ft, lbl)

if __name__ == "__main__":
    with open('train.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, dialect="excel")
        datareader.next()
        data = [row for row in datareader]
        features = preprocessing.scale(np.array([row[1:-1] for row in data]))
        labels = np.array([row[-1][-1] for row in data])

    test_svm(features, labels)

    clf_svm = svm.SVC(gamma=0.01, C=10.)
    clf_svm.fit(features, labels)
    print cross_val_score(clf_svm, features, labels, cv=5)
    #clf_tree = tree.DecisionTreeClassifier()
    #clf_tree.fit(features,labels)

    #scores = cross_val_score(clf_svm,features,labels,cv = 5)
