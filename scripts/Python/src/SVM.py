# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np                     # Llibreria matemÃ tica
import matplotlib.pyplot as plt        # Per mostrar plots
import sklearn                         # Llibreia de DM
import sklearn.datasets as ds            # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv    # Pel Cross-validation
import sklearn.neighbors as nb     
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report     # Per fer servir el knn
 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
 
import numpy as np    # Numeric and matrix computation
import pandas as pd   # Optional: good package for manipulating data
 
def loaddata():
    train = pd.read_csv("BankCleanLearn.csv", sep= ";")
    test = pd.read_csv("BankCleanTest.csv", sep = ";")
    # Separate data from 
    
    X_train=train.drop(['y'], axis=1)
    y_train=train['y']
    X_train.head()
    
    X_test=test.drop(['y'], axis=1)
    y_test=test['y']
    X_test.head()

    ## Transform to numerical dataset
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train.head()
    X_test.head()
    
    
    #Create subset for test without becoming an oldman
    X1_train = X_train.loc[1:9608]
    y1_train = y_train.loc[1:9608]
    
    X2_train = X_train.loc[9608:19216]
    y2_train = y_train.loc[9608:19216]

    X3_train = X_train.loc[19216:28824]
    y3_train = y_train.loc[19216:28824]    
    
    
  
    scaler1 = MinMaxScaler(feature_range=(-1, 1)).fit(X1_train)
    # Apply the normalization trained in training data in both training and test sets
    X1_train = scaler1.transform(X1_train)
    X1_test = scaler1.transform(X_test)
    
    scaler2 = MinMaxScaler(feature_range=(-1, 1)).fit(X2_train)
    # Apply the normalization trained in training data in both training and test sets
    X2_train = scaler2.transform(X2_train)
    X2_test = scaler2.transform(X_test)    


    scaler3 = MinMaxScaler(feature_range=(-1, 1)).fit(X3_train)
    # Apply the normalization trained in training data in both training and test sets
    X3_train = scaler3.transform(X3_train)
    X3_test = scaler3.transform(X_test)    
    
    
    y1_train = y1_train.replace("no", 0)
    y1_train = y1_train.replace("yes", 1)
    
    y2_train = y2_train.replace("no", 0)
    y2_train = y2_train.replace("yes", 1)

    y3_train = y3_train.replace("no", 0)
    y3_train = y3_train.replace("yes", 1)    
    
    y_test = y_test.replace("no", 0)
    y_test = y_test.replace("yes", 1)    
    
    
    
    f_scorer = make_scorer(f1_score,pos_label = 1)

    # List of C values to test. We usualy test diverse orders of magnitude
    #Cs = np.logspace(-3, 11, num=15, base=10.0)
    gammas = [0.000001,0.00001, 0.0001]    
    #Cs = np.logspace(-2, 2, num=5, base=10.0)
    Cs = np.logspace(-1, 1, num=3, base=10.0)
    print("Tested Cs", Cs)
    #param_grid = {'C': Cs}
    param_grid = {'C': Cs, 'gamma' : gammas}
    # fit the model and get the separating hyperplane
    # fit the model and get the separating hyperplane using weighted classes
    
    grid_search = GridSearchCV(SVC(kernel='linear', class_weight="balanced"), param_grid, cv=10,scoring=f_scorer)
    grid_search.fit(X1_train, y1_train)
    print("Best Params",grid_search.best_params_,"F1 score", grid_search.best_score_)
    parval=grid_search.best_params_
    print("Best C =", parval['C'])
    
    clf = SVC(kernel='linear',C=10, class_weight="balanced")
    print("2")
    clf.fit(X1_train,y1_train)
    print("3")
    pred = clf.predict(X1_train)
    print("\n** Results for Plain SVM linear with ratio for class 1 set to 10")
    print(classification_report(y1_train, pred))
    print("Confusion matrix on train set:\n",confusion_matrix(y1_train, pred))    
    
    
    
    
    #QUADRATIC
    grid_search = GridSearchCV(SVC(kernel='quadratic', class_weight="balanced"), param_grid, n_jobs=-1, cv=10,scoring=f_scorer)
    print("1")
    grid_search.fit(X2_train, y2_train)
    print("Best Params",grid_search.best_params_,"F1 score", grid_search.best_score_)
    parval=grid_search.best_params_
    print("Best C =", parval['C'])
    clf = SVC(kernel='quadratic',C=parval['C'],gamma = 0.01, class_weight="balanced")
    print("2")
    clf.fit(X2_train,y2_train)
    print("3")
    pred = clf.predict(X2_train)
    print("\n** Results for Plain SVM Quadratic with ratio for class 1 set to 10")
    print(classification_report(y2_train, pred))
    print("Confusion matrix on train set:\n",confusion_matrix(y2_train, pred))    
    
    
    #RBF
    grid_search = GridSearchCV(SVC(kernel='rbf', class_weight="balanced"), param_grid, n_jobs=-1, cv=10,scoring=f_scorer)
    print("1")
    grid_search.fit(X3_train, y3_train)
    print("Best Params",grid_search.best_params_,"F1 score", grid_search.best_score_)
    parval=grid_search.best_params_
    print("Best C =", parval['C'])
    
    clf = SVC(kernel='rbf',C=10,gamma = 0.01, class_weight="balanced")
    print("2")
    clf.fit(X3_train,y3_train)
    print("3")
    pred = clf.predict(X3_train)
    print("\n** Results for Plain SVM RBF with ratio for class 1 set to 10")
    print(classification_report(y3_train, pred))
    print("Confusion matrix on train set:\n",confusion_matrix(y3_train, pred))    
    
    pred = clf.predict(X3_test)
    print("\n** Results for Plain SVM RBF with ratio for class 1 set to 10")
    print(classification_report(y_test, pred))
    print("Confusion matrix on test set:\n",confusion_matrix(y_test, pred))
    print("Number of supports: ",np.sum(clf.n_support_))
    print("Percentatge of supports: ",np.sum(clf.n_support_)/X3_train.shape[0])    
    
    

loaddata()


