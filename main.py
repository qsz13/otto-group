import csv
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score


with open('train.csv', 'rb') as csvfile:
  datareader = csv.reader(csvfile, dialect="excel")
  datareader.next()
  data = [row for row in datareader]
  features = [row[1:-1] for row in data]
  labels = [row[-1][-1] for row in data]

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(features, labels)

clf1 = tree.DecisionTreeClassifier()
clf1.fit(features,labels)

scores = cross_val_score(clf1,features,labels,cv=5)
print scores
