import csv
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

def get_best_svm_param(ft, lbl):
    #param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}]
    '''
    param_grid = [{'C': [1, 5, 10, 15], 'gamma': [0.1, 0.01, 0.005], 'kernel': ['rbf']}]
    cv = ShuffleSplit(n_splits=4, test_size=0.15, train_size=0.15, random_state=0)
    clf_svm = svm.SVC()
    gs = GridSearchCV(clf_svm, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gs.fit(ft, lbl)
    print gs.best_estimator_, gs.best_params_, gs.best_score_
    best_param = gs.best_params_
    '''
    print 'Start on the whole training set...'
    #clf_svm = svm.SVC(kernel = best_param['kernel'], gamma = best_param['gamma'], C = best_param['C'])
    clf_svm = svm.SVC(kernel='rbf', gamma=0.01, C=5)
    clf_svm.fit(features, labels)
    print cross_val_score(clf_svm, features, labels, cv = 5)

if __name__ == "__main__":

    #with open('train.csv', 'rb') as csvfile:
    #    datareader = csv.reader(csvfile, dialect="excel")
    #    datareader.next()
    #    data = [row for row in datareader]
        #features = preprocessing.scale(np.array([row[1:-1] for row in data]))
    #    features = np.array([row[1:-1] for row in data])
    #    labels = np.array([row[-1][-1] for row in data])

    # Import Data
    tests = pd.read_csv('test.csv')
    tests = tests.drop('id', axis=1)
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    tests = scaler.fit_transform(tests)
    #print tests
    
    #features = pd.read_csv('../input/train.csv')
    features = pd.read_csv('train.csv')
    features = features.drop('id', axis=1)
    
    # Extract target and Encode it to make it manageable by ML algo
    labels = features.target.values
    labels = LabelEncoder().fit_transform(labels)
    #print labels
    
    # Remove target from train, else it's too easy ...
    features = features.drop('target', axis=1)

    #features = preprocessing.normalize(features)
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    features = scaler.fit_transform(features, labels)
    #print features
    #minmax scalar
    #min_max_scaler = preprocessing.MinMaxScaler()
    #features = min_max_scaler.fit_transform(features)
    #print features

    # svm 81.5%
    #get_best_svm_param(features, labels)


    # decision tree 70%
    #clf_tree = tree.DecisionTreeClassifier()
    #clf_tree.fit(features,labels)
    #scores = cross_val_score(clf_svm,features,labels,cv = 5)


    # Split Train / Test (needed for calibrated random forest)
    #features, features_test, labels, labels_test = train_test_split(features, labels, test_size=0.20, random_state=36)

    # calibrated random forest 81.6%
    '''
    clf = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    calibrated_clf.fit(features, labels)
    predicted_labels = calibrated_clf.predict(tests)
    len_lbl = len(predicted_labels)
    #print cross_val_score(calibrated_clf, features, labels, cv=5)
    #ypreds = calibrated_clf.predict_proba(features_test)
    #print("logloss with calibration : ", log_loss(labels_test, ypreds, eps=1e-15, normalize=True))
    '''
    
    dtrain = xgb.DMatrix(features, label=labels)
    param = {'eta':0.05,'min_child_weight':5.5,'max_delta_step':0.45,'max_depth':12,'silent':1, 'objective':'multi:softprob', 'nthread':60, 'eval_metric':'mlogloss','num_class':9,'subsample':1,'colsample_bytree':0.5,'gamma':0.5}
    num_round = 900
    bst = xgb.train(param, dtrain, num_round)
    print cross_val_score(calibrated_clf, features, labels, cv=5)
    
    predicted_labels = bst.predict(tests)
    len_lbl = len(predicted_labels)
    # knn 78%
    #neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    #neigh.fit(features, labels)
    #print cross_val_score(neigh, features, labels, cv=5)

    # MLP 79%
    #clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (25,), random_state = 1)
    #clf_mlp.fit(features, labels)
    #print cross_val_score(clf_mlp, features, labels, cv=5)
    
    output_mtx = np.zeros((len_lbl,10),dtype=np.uint32)
    for i in xrange(len_lbl):
        output_mtx[i,0] = i + 1
        output_mtx[i,predicted_labels[i]+1] = 1
    #print output_mtx[0]
    #print output_mtx[144367]
    raw_data = {'id': output_mtx[:,0],
        'class_1': output_mtx[:,1],
        'class_2': output_mtx[:,2],
        'class_3': output_mtx[:,3],
        'class_4': output_mtx[:,4],
        'class_5': output_mtx[:,5],
        'class_6': output_mtx[:,6],
        'class_7': output_mtx[:,7],
        'class_8': output_mtx[:,8],
        'class_9': output_mtx[:,9]
        }
    df = pd.DataFrame(raw_data, columns = ['id', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9'])
    df.to_csv('result.csv',index = False)
    