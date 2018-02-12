# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:43:47 2018

@author: josec
"""

##################################################################################################################################################################################################
# All libraries here
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
##################################################################################################################################################################################################
# Create the path variable
pathIn = '/udacity-data/'
pathOut = '/output/'
##################################################################################################################################################################################################
# Import the train and test data
train = pd.read_csv(pathIn+'train.csv', na_values=-1)
test = pd.read_csv(pathIn+'test.csv', na_values=-1)
##################################################################################################################################################################################################
# Sort the rows by the id
train = train.sort_values(['id'], ascending=True)
test = test.sort_values(['id'], ascending=True)
##################################################################################################################################################################################################
# Features pre treatment
toRemove =['ps_car_05_cat', 'ps_car_03_cat', 'ps_ind_02_cat']
train = train.drop(toRemove, axis=1)
test = test.drop(toRemove, axis=1)
##################################################################################################################################################################################################
# Predicting features
def predictFeatures(model, data, dataMissing, column, op=False):
        
    X_train = data.drop(['id', column], axis=1)
    y_pred = data[column]
    
    if op:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train) 
    
    model.fit(X_train, y_pred)
    
    X_pred = dataMissing.drop(['id', column], axis=1)
    
    if op:
        X_pred = sc.transform(X_pred)
    
    y_pred = model.predict(X_pred)
    
    dataMissing[column] = y_pred
    
    data = data.append(dataMissing)
    
    return data
##################################################################################################################################################################################################
# Treating the features with missings
def missingFeatures(data):
    
    cols_with_missing_values = data.columns[data.isnull().any()].tolist()
    cols_without_missing_values = data.columns[data.isnull().any()==0].tolist()
    
    cat_cols_without_missing_values = [c for c in cols_without_missing_values if ('cat' in c) or ('id' in c)] 
    bin_cols_without_missing_values = [c for c in cols_without_missing_values if ('bin' in c)]
    other_cols_without_missing_values = [c for c in cols_without_missing_values if ('bin' not in c and 'cat' not in c and 'target' not in c and 'id' not in c)]
    
    for column in data.columns:
        
        if column != 'id' and column != 'target':
            if column in cols_with_missing_values:
                print('##################################################################')
                print('Starting treatment of:', column)
                if 'cat' in column:
                    print('Missings:', data[column].isnull().sum())
                    cols = cat_cols_without_missing_values
                    cols.append(column)                    
                    toTrain = data[cols][(data[column].notnull())]  
                    toPredict = data[cols][(data[column].isnull())]
                    newColumn = predictFeatures(LogisticRegression(n_jobs=-1, random_state=0), toTrain, toPredict, column, True)
                    newColumn = newColumn.sort_values(['id'], ascending=True)
                    data[column] = newColumn[column]
                    print ('Finished')
                    print ('Missings: ', data[column].isnull().sum())
                    print ('##################################################################')
                           
                elif 'reg' in column:
                    print('Missings:', data[column].isnull().sum())
                    toTrain = data[['id','ps_reg_02','ps_reg_03']][(data['ps_reg_03'].notnull())]         
                    toPredict = data[['id','ps_reg_02','ps_reg_03']][(data['ps_reg_03'].isnull())]
                    newColumn = predictFeatures(LinearRegression(n_jobs=-1), toTrain, toPredict, column, False)
                    newColumn = newColumn.sort_values(['id'], ascending=True)
                    data[column] = newColumn[column]
                    print ('Finished')
                    print ('Missings: ', data[column].isnull().sum())
                    print ('##################################################################')
                
                elif 'car_12' in column:
                    print('Missings:', data[column].isnull().sum())
                    toTrain = data[['id','ps_car_13','ps_car_12']][(data['ps_car_12'].notnull())]       
                    toPredict= data[['id','ps_car_13','ps_car_12']][(data['ps_car_12'].isnull())]                    
                    newColumn = predictFeatures(LinearRegression(n_jobs=-1), toTrain, toPredict, column, True)
                    newColumn = newColumn.sort_values(['id'], ascending=True)
                    data[column] = newColumn[column]
                    print ('Finished')
                    print ('Missings: ', data[column].isnull().sum())
                    print ('##################################################################')
                           
                else:
                    print('Missings:', data[column].isnull().sum())
                    data[column].fillna(data[column].mean(), inplace=True)
                    print ('Finished')
                    print ('Missings: ', data[column].isnull().sum())
                    print ('##################################################################')
                    
        else:
            continue
                    
    return data
##################################################################################################################################################################################################
# Taking care of the missing data
train = missingFeatures(train)
test = missingFeatures(test)
##################################################################################################################################################################################################
# Taking care of calculated features
features = train.columns.tolist()
features = [c for c in features if 'target' not in c and 'id' not in c]

features_cat = [c for c in features if 'cat' in c]
features_bin = [c for c in features if 'bin' in c]
features_num = [c for c in features if c not in features_bin and c not in features_cat]
features_calc = [c for c in features_num if 'calc' in c]
features_reg = [c for c in features_num if 'reg' in c]
features_car = [c for c in features_num if 'car' in c]

train['new_calc'] = 1
test['new_calc'] = 1
for f in features_calc:
    train['new_calc'] = train['new_calc'] + train[f]
    test['new_calc'] = test['new_calc'] + test[f]
    
train = train.drop(features_calc, axis=1)
test = test.drop(features_calc, axis=1)
##################################################################################################################################################################################################
# Cat features
for f in features_cat:
    le = LabelEncoder()
    le.fit(train[f])
    train[f] = le.fit_transform(train[f])
    test[f] = le.fit_transform(test[f])
##################################################################################################################################################################################################
# Submission
def submissionCreate(data, test, model, name, path):
    # Start to count the time
    start = time()
    
    
    # Creating X and y
    X = data.drop(['target', 'id'], axis=1)
    y = data['target']   
    
    
    print('Train starting...')
    
    model.fit(X, y)
        
    id_test = test['id'].values
    
    test_pred = test.drop(['id'], axis=1)
    
    submission = pd.DataFrame(columns=['id','target'])
    submission['target'] = model.predict_proba(test_pred)[:,1]
    submission['id'] = id_test
    submission.to_csv(path+name, index=False)
    
    # Stop the time count
    end = time()
    
    # Print the time
    print ("Time: {:.4f} seconds".format(end - start))

##################################################################################################################################################################################################
# XGB1

print('##################################################################')    
print('Creating XGB')

params = {}
params['learning_rate'] = 0.02
params['n_estimators'] = 1000
params['max_depth'] = 4
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9
params['n_jobs'] = 8


clf = XGBClassifier(**params)
submissionCreate(train, test, clf, 'submission_final.csv', pathOut)
print('##################################################################') 