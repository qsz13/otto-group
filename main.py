import csv
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np

def get_best_svm_param(ft, lbl):
    param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
    cv = ShuffleSplit(n_splits=4, test_size=0.15, train_size=0.15, random_state=0)
    clf_svm = svm.SVC()
    gs = GridSearchCV(clf_svm, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gs.fit(ft, lbl)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

if __name__ == "__main__":
    with open('train.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, dialect="excel")
        datareader.next()
        data = [row for row in datareader]
        features = preprocessing.scale(np.array([row[1:-1] for row in data]))
        labels = np.array([row[-1][-1] for row in data])

    results = get_best_svm_param(features, labels)
    print results
    best_param = results[1]

    print 'Start on the whole training set...'
    clf_svm = svm.SVC(kernel = best_param['kernel'], gamma = best_param['gamma'], C = best_param['C'])
    clf_svm.fit(features, labels)
    print cross_val_score(clf_svm, features, labels, cv = 5)
    #clf_tree = tree.DecisionTreeClassifier()
    #clf_tree.fit(features,labels)

    #scores = cross_val_score(clf_svm,features,labels,cv = 5)
