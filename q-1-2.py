import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

data = pd.read_csv("data.csv", header=None)
data.columns = ['ID','Age','Experience','Annual_Income','ZIP_Code','Family_size','Avg_Expenditure_pm','Education_Level','Mortgage_value','Loan_Accepted','Have_Security','Have_CD','Have_NetBanking','Have_Credit_Card']

X_train, X_validation, y_train, y_validation = train_test_split(
    data,
    data[['Loan_Accepted']],
    test_size=0.2,
    random_state=0)


categorical_columns = ['ZIP_Code','Education_Level','Have_Security','Have_CD','Have_NetBanking','Have_Credit_Card']
numeric_columns = ['Age','Experience','Annual_Income','Family_size','Avg_Expenditure_pm','Mortgage_value']
freq_vals = dict()
for cat_col in categorical_columns:
    freq_vals[cat_col] = X_train[cat_col].mode()[0]


def compute_gaussian_prob(val, mean, std):
    p = (1.0/np.sqrt(2*np.pi*std*std))*np.exp(-(float((val-mean)**2)/float(2*std*std)))
    return p



class NaiveBayes:
    P = None
    
    def compute_accuracy(self,y_actual, y_predict):
        y_actual = list(y_actual)
        y_predict = list(y_predict)
        hits = 0
        for i in range(0,len(y_actual)):
            if y_actual[i] == y_predict[i]:
                hits+=1
        return float(hits)/float(len(y_actual))
    
    def add_category_data(self, col, category, cat_accepted, cat_not_accepted, total_accepted, total_not_accepted):
        if 'category' not in self.P:
            self.P['category'] = dict()
        if col not in self.P['category']:
            self.P['category'][col] = dict()
        self.P['category'][col][category] = dict()
        self.P['category'][col][category][1] = float(cat_accepted)/float(total_accepted)
        self.P['category'][col][category][0] = float(cat_not_accepted)/float(total_not_accepted)
        
    def add_numeric_data(self, col, acc_mean, not_acc_mean, acc_std, not_acc_std):
        if 'numeric' not in self.P:
            self.P['numeric'] = dict()
        if col not in self.P['numeric']:
            self.P['numeric'][col] = dict()
        self.P['numeric'][col][1] = dict()
        self.P['numeric'][col][0] = dict()
        self.P['numeric'][col][1]['mean'] = acc_mean
        self.P['numeric'][col][0]['mean'] = not_acc_mean
        self.P['numeric'][col][1]['std'] = acc_std
        self.P['numeric'][col][0]['std'] = not_acc_std
    
    def predict(self, X_test, cat_cols, num_cols):
        y_pred = list()
        p_pos = 0
        p_neg = 0
        for index,row in X_test.iterrows():
            for col in X_test:
                val = row[col]
                if col in cat_cols:
                    if val not in self.P['category'][col]:
                        val = freq_vals[col]
                    p_pos += 0 if self.P['category'][col][val][1] == 0 else np.log(self.P['category'][col][val][1])
                    p_neg += 0 if self.P['category'][col][val][0] == 0 else np.log(self.P['category'][col][val][0])
                if col in num_cols:
                    acc_mean = self.P['numeric'][col][1]['mean']
                    acc_std = self.P['numeric'][col][1]['std']
                    nacc_mean = self.P['numeric'][col][0]['mean']
                    nacc_std = self.P['numeric'][col][0]['std']
                    p_pos += np.log(compute_gaussian_prob(val,acc_mean,acc_std))
                    p_neg += np.log(compute_gaussian_prob(val,nacc_mean,nacc_std))
            p_pos += np.log(self.P[1])
            p_neg += np.log(self.P[0])
            if p_pos > p_neg:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
                
    
    def train(self, X_train, cat_cols, num_cols):
        
        total_accepted = len(X_train.loc[X_train['Loan_Accepted']==1])
        total_not_accepted = len(X_train.loc[X_train['Loan_Accepted']==0])
        total = len(X_train)
        self.P = dict()
        self.P[1] = float(total_accepted)/float(total)
        self.P[0] = float(total_not_accepted)/float(total)
        
        for col in X_train:
            if col in cat_cols:
                categories = X_train[col].unique()
                total_pos = 0
                total_neg = 0
                for cat in categories:
                    if 1 in X_train.groupby([col])['Loan_Accepted'].value_counts()[cat]:
                        total_pos += X_train.groupby([col])['Loan_Accepted'].value_counts()[cat][1]
                    if 0 in X_train.groupby([col])['Loan_Accepted'].value_counts()[cat]:
                        total_neg += X_train.groupby([col])['Loan_Accepted'].value_counts()[cat][0]
                for cat in categories:
                    pos = 0
                    neg = 0
                    if 1 in X_train.groupby([col])['Loan_Accepted'].value_counts()[cat]:
                        pos = X_train.groupby([col])['Loan_Accepted'].value_counts()[cat][1]
                    if 0 in X_train.groupby([col])['Loan_Accepted'].value_counts()[cat]:
                        neg = X_train.groupby([col])['Loan_Accepted'].value_counts()[cat][0]
                    self.add_category_data(col,cat,pos,neg,total_pos,total_neg)
            if col in num_cols:
                acc_mean = X_train.loc[X_train['Loan_Accepted']==1][col].mean()
                not_acc_mean = X_train.loc[X_train['Loan_Accepted']==0][col].mean()
                acc_std = X_train.loc[X_train['Loan_Accepted']==1][col].std()
                not_acc_std = X_train.loc[X_train['Loan_Accepted']==0][col].std()
                self.add_numeric_data(col, acc_mean, not_acc_mean, acc_std, not_acc_std)
        return self.P



nb = NaiveBayes()
p = nb.train(X_train, categorical_columns, numeric_columns)
y_train_pred = nb.predict(X_train, categorical_columns, numeric_columns)
y_valid_pred = nb.predict(X_validation, categorical_columns, numeric_columns)

train_acc = nb.compute_accuracy(X_train['Loan_Accepted'],y_train_pred)
valid_acc = nb.compute_accuracy(X_validation['Loan_Accepted'], y_valid_pred)

test_file_name = sys.argv[1]
X_test = pd.read_csv(test_file_name, header=None)
X_test.columns = ['ID','Age','Experience','Annual_Income','ZIP_Code','Family_size','Avg_Expenditure_pm','Education_Level','Mortgage_value','Loan_Accepted','Have_Security','Have_CD','Have_NetBanking','Have_Credit_Card']

y_test_pred = nb.predict(X_test, categorical_columns, numeric_columns)
test_acc = nb.compute_accuracy(X_test['Loan_Accepted'], y_test_pred)



print '*****************BANK DATASET*****************'
print 'Train Accuracy : '+str(train_acc*100)
print 'Validation Accuracy : '+str(valid_acc*100)
print
print 'Input test file accuracy : '+str(test_acc*100)
print '**********************************************'





