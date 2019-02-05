import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import train_test_split
import sys


data = pd.read_csv("Iris.csv", header=None)
data.columns = ['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm','class']
data['Id'] = range(1, len(data) + 1)

X_train, X_validation, y_train, y_validation = train_test_split(
    data,
    data[['class']],
    test_size=0.2,
    random_state=0)

class KNearestNeighbors:
    def compute_accuracy(self,y_actual, y_predict):
        y_actual = list(y_actual)
        y_predict = list(y_predict)
        hits = 0
        for i in range(0,len(y_actual)):
            if y_actual[i] == y_predict[i]:
                hits+=1
        return float(hits)/float(len(y_actual))
    def predict(self,train_data, predict_data, k):
        dist_dict = dict()
        id_class_dict = dict()
        inc_cols = ['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm']
        predict_class = list()
        for predict_index,predict_row in predict_data.iterrows():
            for train_index,train_row in train_data.iterrows():
                if train_row['Id']==predict_row['Id']:
                    continue
                d = np.sqrt(np.sum(np.square(np.subtract(np.array(train_row[inc_cols]),np.array(predict_row[inc_cols])))))
                dist_dict[train_row['Id']] = d
                id_class_dict[train_row['Id']] = train_row['class']
            sorted_dist = sorted(dist_dict.items(), key=operator.itemgetter(1))[:k]
            class_dict = dict()
            for d in sorted_dist:
                c = id_class_dict[d[0]]
                if c in class_dict:
                    class_dict[c] += 1
                else:
                    class_dict[c] = 1
            max_class = max(class_dict, key=class_dict.get)
            predict_class.append(max_class)
        return predict_class

knn = KNearestNeighbors()
train_pred = knn.predict(X_train,X_train,9)
train_acc = knn.compute_accuracy(X_train['class'],train_pred)
    
valid_pred = knn.predict(X_train,X_validation,9)
valid_acc = knn.compute_accuracy(X_validation['class'],valid_pred)

test_file_name = sys.argv[1]
X_test = pd.read_csv(test_file_name, header=None)
X_test.columns = ['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm','class']
X_test['Id'] = range(1, len(X_test) + 1)

test_pred = knn.predict(X_train,X_test,9)
test_acc = knn.compute_accuracy(X_test['class'],test_pred)


print '**************Iris Dataset******************'
print 'K : 9'
print 'Train Accuracy : '+str(train_acc*100)
print 'Validation Accuracy : '+str(valid_acc*100)
print
print 'Input test file accuracy : '+str(test_acc*100)
print '**********************************************'

