#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression on heart dataset
"""

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

raw = loadmat('heart.mat')

data = raw['dat']
label = raw['label'].ravel()

Xtrain, Xtest, ytrain, ytest = train_test_split(data, label, test_size=0.2, random_state=1)

clf = LogisticRegression(max_iter=200, solver='liblinear').fit(Xtrain, ytrain)

# ## training error
ypred_train = clf.predict(Xtrain)
matched_train = ypred_train == ytrain
acc_train = sum(matched_train)/len(matched_train)

# training confusion matrix
idx1 = np.where(ytrain ==1) 
idx2 = np.where(ytrain ==2)
cf_train_11 = np.sum(ytrain[idx1] == ypred_train[idx1])
cf_train_12 = np.sum(ypred_train[idx1] ==2)
cf_train_22 = np.sum(ytrain[idx2] == ypred_train[idx2])
cf_train_21 = np.sum(ypred_train[idx2] ==1)

# ## test error
ypred_test = clf.predict(Xtest)
matched_test = ypred_test == ytest
acc_test = sum(matched_test)/len(matched_test)

# testing confusion matrix
idx1 = np.where(ytest ==1) 
idx2 = np.where(ytest ==2)
cf_test_11 = np.sum(ytest[idx1] == ypred_test[idx1])
cf_test_12 = np.sum(ypred_test[idx1] ==2)
cf_test_22 = np.sum(ytest[idx2] == ypred_test[idx2])
cf_test_21 = np.sum(ypred_test[idx2] ==1)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('the training accuracy: '+ str(round(acc_train, 4)))
print('confusion matrix for training:')
print('          predected 1       predected 2')
print(f"true 1        {cf_train_11}                {cf_train_12}")
print(f"true 2        {cf_train_21}                {cf_train_22}")
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('the testing accuracy: '+ str(round(acc_test, 4)))
print('confusion matrix for testing:')
print('          predected 1       predected 2')
print(f"true 1        {cf_test_11}                {cf_test_12}")
print(f"true 2        {cf_test_21}                {cf_test_22}")
