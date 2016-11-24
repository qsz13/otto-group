import csv
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import numpy as np

def get_best_svm_param(ft, lbl):
    #param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}]
    param_grid = [{'C': [1, 5, 10, 15], 'gamma': [0.1, 0.01, 0.005, 0.001], 'kernel': ['rbf']}]
    cv = ShuffleSplit(n_splits=4, test_size=0.15, train_size=0.15, random_state=0)
    clf_svm = svm.SVC()
    gs = GridSearchCV(clf_svm, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gs.fit(ft, lbl)
    print gs.best_estimator_, gs.best_params_, gs.best_score_
    best_param = gs.best_params_
    print 'Start on the whole training set...'
    clf_svm = svm.SVC(kernel = best_param['kernel'], gamma = best_param['gamma'], C = best_param['C'])
    clf_svm.fit(features, labels)
    print cross_val_score(clf_svm, features, labels, cv = 5)

if __name__ == "__main__":
    '''
    with open('train.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, dialect="excel")
        datareader.next()
        data = [row for row in datareader]
        #features = preprocessing.scale(np.array([row[1:-1] for row in data]))
        features = np.array([row[1:-1] for row in data])
        labels = np.array([row[-1][-1] for row in data])
    '''
    # Import Data
    #features = pd.read_csv('../input/train.csv')
    features = pd.read_csv('train.csv')
    features = features.drop('id', axis=1)

    # Extract target
    # Encode it to make it manageable by ML algo
    labels = features.target.values
    labels = LabelEncoder().fit_transform(labels)

    # Remove target from train, else it's too easy ...
    features = features.drop('target', axis=1)

    #svm
    #get_best_svm_param(features, labels)


    #decision tree
    #clf_tree = tree.DecisionTreeClassifier()
    #clf_tree.fit(features,labels)
    #scores = cross_val_score(clf_svm,features,labels,cv = 5)

    # Split Train / Test
    features, features_test, labels, labels_test = train_test_split(features, labels, test_size=0.20, random_state=36)

    #calibrated random forest
    clf = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    calibrated_clf.fit(features, labels)
    ypreds = calibrated_clf.predict_proba(features_test)
    print("logloss with calibration : ", log_loss(labels_test, ypreds, eps=1e-15, normalize=True))
    print cross_val_score(calibrated_clf, features, labels, cv=5)
