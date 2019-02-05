import pandas as pd
import numpy as np
import operator
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("admission_data.csv")


X_train, X_validation, y_train, y_validation = train_test_split(
    data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']],
    data[['Chance of Admit']],
    test_size=0.2,
    random_state=0)


test_input_file = sys.argv[1]
test_data = pd.read_csv(test_input_file)
X_test = test_data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y_test = test_data[['Chance of Admit']]


for col in X_train:
    mean = X_train[col].mean()
    std = X_train[col].std()
    X_train[col] = (X_train[col] - mean)/std
    X_validation[col] = (X_validation[col]-mean)/std
    X_test[col] = (X_test[col]-mean)/std


X_train['Ones'] = [1]*len(X_train)
X_validation['Ones'] = [1]*len(X_validation)
X_test['Ones'] = [1]*len(X_test)

class LinearRegression:
    theta = None
    
    def predict(self, X):
        Y_pred = np.dot(X.values,self.theta.T)
        return Y_pred
    
    def compute_error(self, y_pred, y_actual, error_function):
        m = len(y_actual)
        if error_function == 'mean_squared_error':
            error = (1.0/float(m))*np.sum((y_pred-y_actual)*(y_pred-y_actual))
        elif error_function == 'mean_absolute_error':
            error = (1.0/float(m))*np.sum(np.absolute(y_pred-y_actual))
        elif error_function == 'mean_absolute_percentage_error':
            error = (1.0/float(m))*np.sum(np.absolute(np.divide((y_pred-y_actual),y_actual)))
        return error
    
    def compute_gradient(self, X, h, Y, error_function):
        m = len(Y)
        if error_function == 'mean_squared_error':
            grad = (2.0/float(m))*np.sum(X*(h-Y), axis=0)
        elif error_function == 'mean_absolute_error':
            grad = (1.0/float(m))*np.sum(X*np.divide(h-Y, np.absolute(Y-h)),axis=0)
        elif error_function == 'mean_absolute_percentage_error':
            grad = (1.0/float(m))*np.sum(X*np.divide((h-Y),(Y*Y*np.absolute(np.divide(h,Y)-1.0))),axis=0)
        return grad
    
    def train(self, X_train, y_train, alpha, max_epochs,error_function):
        self.theta = None
        self.theta = np.random.rand(1,X_train.shape[1])
        m = len(X_train)
        its = list()
        errs = list()
        for i in range(0,max_epochs):
            X = X_train.values
            h = np.dot(X,self.theta.T)
            Y = y_train.values
            self.theta = self.theta - alpha*self.compute_gradient(X,h,Y,error_function)
            
            y_pred = np.dot(X,self.theta.T)
            err = self.compute_error(y_pred,y_train.values,error_function)
            its.append((i+1))
            errs.append(err)
        return (its,errs)


ln_mse = LinearRegression()
its_mse,errs_mse = ln_mse.train(X_train,y_train,0.1, 100,'mean_squared_error')
y_train_pred_mse = ln_mse.predict(X_train)
y_valid_pred_mse = ln_mse.predict(X_validation)
y_test_pred_mse = ln_mse.predict(X_test)

train_mse = ln_mse.compute_error(y_train_pred_mse, y_train.values,'mean_squared_error')
valid_mse = ln_mse.compute_error(y_valid_pred_mse, y_validation.values,'mean_squared_error')
test_mse = ln_mse.compute_error(y_test_pred_mse, y_test.values,'mean_squared_error')
print '*****************MEAN SQUARED ERROR*****************'
print
print 'Train set error : '+str(train_mse)
print 'Validation set error : '+str(valid_mse)
print
print 'Input test file error : '+str(test_mse)
print
print '****************************************************'
print

ln_mae = LinearRegression()
its_mae,errs_mae = ln_mae.train(X_train,y_train,0.001, 3000,'mean_absolute_error')
y_train_pred_mae = ln_mae.predict(X_train)
y_valid_pred_mae = ln_mae.predict(X_validation)
y_test_pred_mae = ln_mae.predict(X_test)

train_mae = ln_mae.compute_error(y_train_pred_mae, y_train.values,'mean_absolute_error')
valid_mae = ln_mae.compute_error(y_valid_pred_mae, y_validation.values,'mean_absolute_error')
test_mae = ln_mae.compute_error(y_test_pred_mae, y_test.values,'mean_absolute_error')
print '*****************MEAN ABSOLUTE ERROR*****************'
print
print 'Train set error : '+str(train_mae)
print 'Test set error : '+str(valid_mae)
print
print 'Input test file error : '+str(test_mae)
print
print '****************************************************'
print

ln_mape = LinearRegression()
its_mape,errs_mape = ln_mape.train(X_train,y_train,0.01, 250,'mean_absolute_percentage_error')
y_train_pred_mape = ln_mape.predict(X_train)
y_valid_pred_mape = ln_mape.predict(X_validation)
y_test_pred_mape = ln_mape.predict(X_test)

train_mape = ln_mape.compute_error(y_train_pred_mape, y_train.values,'mean_absolute_percentage_error')
valid_mape = ln_mape.compute_error(y_valid_pred_mape, y_validation.values,'mean_absolute_percentage_error')
test_mape = ln_mape.compute_error(y_test_pred_mape, y_test.values,'mean_absolute_percentage_error')
print '*****************MEAN ABSOLUTE PERCENTAGE ERROR*****************'
print
print 'Train set error : '+str(train_mape)
print 'Test set error : '+str(valid_mape)
print
print 'Input test file error : '+str(test_mape)
print
print '****************************************************'
print

