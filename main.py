import csv
from sklearn import svm
from sklearn.model_selection import cross_val_score


with open('train.csv', 'rb') as csvfile:
  datareader = csv.reader(csvfile, dialect="excel")
  datareader.next()
  data = [row for row in datareader]
  features = [row[:-1] for row in data]
  labels = [row[-1][-1] for row in data]
  
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(features[:-1], labels[:-1])

scores = cross_val_score(clf,features,labels,cv=5)
print scores
