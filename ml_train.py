# -*- coding: utf-8 -*-

# Test out various machine learning techniques

import numpy as np
import pandas as pd
from Clean_df import Clean_df

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

load_folder = 'ml_model'
save_folder = 'ml_train'

clean = Clean_df(load_folder, save_folder)

# Variable to predict is the rating
# Two models, one for Seoul ratings and one for Nationwide ratings
# Average between TNmS and AGB if both present, else take either

'----- Load Data, Create features, label split -----'

data = clean.load_df('data')

y = data['rating']
print('y: ', y)

X = data.drop(columns=['rating', 'drama'])
print('X: ', X.info())

# Group columns into dtypes
col_by_types = X.columns.groupby(X.dtypes)

for k in col_by_types.keys():
    print('dtype: ', k)
    print('len: ', len(col_by_types[k]))
    print('columns: ', col_by_types[k])
    
# Get all variables that are not bool type
non_bool_vars = []

for k in col_by_types.keys():
    if k != 'bool':
        non_bool_vars += col_by_types[k].tolist()
        
print('No. of non-bool variables: ', len(non_bool_vars))
print('Non-bool variables: ', non_bool_vars)

# Check if they have multiple labels
# Convert to list if commas present

# ohe for single labels
# mlb for multiple labels

ohe_list = []
mlb_list = []

for var in non_bool_vars:
    if X[var].dtype == 'object':
        X[var] = X[var].apply(lambda x: x.split(','))
        if X[var].apply(len).max() == 1:
            X[var] = X[var].apply(lambda x: x[0])
            ohe_list.append(var)
        else:
            mlb_list.append(var)
    else:
        ohe_list.append(var)

for var in ohe_list:
    print('le: ', X[var])
    
for var in mlb_list:
    print('mlb: ', X[var])

# Convert all NaN to zeros
X = X.fillna(0)

'Try encoding with a few methods: Ordinal, Binary, One-Hot, Hashing'
'Hashing allows for new categories'

# from sklearn.preprocessing import OrdinalEncoder
# ordinalenc = OrdinalEncoder()
# # fit(X) categories_ transform(X)

# from sklearn.feature_extraction import FeatureHasher
# hasher = FeatureHasher(n_features=10)
# feature = hasher.transform(df).toarray()

# To import the different methods and then use each one.
# Maybe convert the process code into a function

enc = None

from sklearn.preprocessing import OrdinalEncoder, \
                                  OneHotEncoder, \
                                  MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher

def encode(df, var_list=None, method='ordinal', n_features=10):
    # methods: ordinal, onehot, hash
    # get df, var_list, and method
    # returns new df with encoded vars
    
    if var_list==None:
        var_list = df.columns
    
    if method == 'ordinal':
        enc = OrdinalEncoder()
        encoded = enc.fit_transform(df[var_list]).toarray()
        labels = enc.categories_
        labels = np.hstack(labels).ravel()
        return pd.DataFrame(encoded, columns=labels).astype(bool)
        
    elif method == 'onehot':
        enc = OneHotEncoder()
        encoded = enc.fit_transform(df[var_list]).toarray()
        labels = enc.categories_
        labels = np.hstack(labels).ravel()
        return pd.DataFrame(encoded, columns=labels).astype(bool)
        
    elif method == 'hash':
        df = pd.DataFrame()
        for var in var_list:
            enc = FeatureHasher(n_features=n_features,
                            input_type='string')
            encoded = enc.transform(df[var_list]).toarray()
            labels = [var_list+'_'+str(x) for x in range(n_features)]
            var_df = pd.DataFrame(encoded, columns=labels).astype(bool)
            df = pd.concat([df,var_df], axis=1)
        return df
    
    elif method == 'mlb':
        df = pd.DataFrame()
        for var in var_list:
            enc = MultiLabelBinarizer()
            encoded = enc.fit_transform(X[var])
            labels = enc.classes_
            var_df = pd.DataFrame(encoded, columns=labels).astype(bool)
            df = pd.concat([df, var_df], axis=1)
        return df
            
'Getting the encoding done with OneHotEncoder and MultiLabelBinarizer'
# Create df with type columns
ohe_df = encode(X,
                ohe_list,
                method='onehot')

mlb_df = encode(X,
                mlb_list,
                method='mlb')

# print('mlb', mlb_df)

# print('BEFORE MERGING: ')
# print(X.info())

# print('Checking ohe_df: ')
# print(ohe_df.info())

# print('Checking mlb_df: ')
# print(mlb_df.info())

'Drop pre-encoded data and merging both encoded dfs into X'
# Remove unencoded columns
X = X.drop(columns=non_bool_vars)

# print('AFTER DROPPING: ')
# print(X.info())

# Merge encoded dfs
X = X.join(ohe_df)
X = X.join(mlb_df)

# print('ARTER MERGING: ')
# print(X.info())

# print(X.info())

# Group columns into dtypes
# col_by_types = X.columns.groupby(X.dtypes)

# for k in col_by_types.keys():
#     print('dtype: ', k)
#     print('columns: ', col_by_types[k])

'----- train_test_split -----'

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=100)

'----- Simple mean prediction -----'

print()
print('Simple mean prediction')

#print(y_train.mean())
y_pred = [y_train.mean()] * len(X_test)

print('r2_score: ', r2_score(y_test, y_pred))

'----- Linear regression prediction -----'

print()
print('== Linear regression prediction ==')

# Drop all columns with null values
X_train_non_nulls = X_train.dropna(axis=1)

from sklearn.linear_model import LinearRegression

cv_score = cross_val_score(LinearRegression(), 
                           X_train_non_nulls, y_train,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1,
                           )
print('cv_score: ', cv_score)
print('cv_score (average): ', cv_score.mean())




'Which metric to use?'
'Feature selection: SelectKBest()'





'----- sklearn methods -----'

# What are the different methods to try?
# https://scikit-learn.org/stable/supervised_learning.html
# Using hash on cast?

# Use CVS instead of fit/predict

from sklearn.model_selection import GridSearchCV

def gridscv(model, parameters, X_train, y_train, scoring='r2', cv=5, n_jobs=-1, verbose=1):
    search = GridSearchCV(model,
                          param_grid=parameters,
                          scoring=scoring,
                          cv=cv,
                          n_jobs=n_jobs,
                          verbose=verbose,
                          )
    search.fit(X_train, y_train)
    return search.cv_results_

'''    
# SVM Regression
from sklearn import svm
print('== Support Vector Machine prediction==')

# Vary the C and epsilon parameters in the model
parameters = {'C':[1**x for x in range(-1, 2)],
              'epsilon':[e*0.1 for e in range(1,5)],
              'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
              'degree':[d for d in range(2,4)],
              }
              
model = svm.SVR()
cv_results = gridscv(model,
                     parameters,
                     X_train_non_nulls,
                     y_train,
                     )
print('cv_results: ', cv_results)

np.savez_compressed('svm_results.npz', cv_results)
'''

# Loading npz file
svm_results = np.load('svm_results.npz',
                      allow_pickle=True,
                      )['arr_0']

print('svm_results: ', svm_results)







for x in np.nditer(svm_results):
    print(x)
print(type(x))

'not working.. use pprint? or switch save/load method to csv'






# ** which kernel to use?

# Stochastic Gradient Descent Regression
# Not too good for low number of training examples
# from sklearn.linear__model import SGDRegressor
# model = SGDRegressor()
# ** May need to tweak learning rate

# Nearest Neighbors Regression
# Either KNeighborsRegressor or RadiusNeighborsRegressor
# from sklearn.neighbors import KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=5)
# from sklearn.beighbors import RadiusNeighborsRegressor
# model.predict(Xtest)

# Decision Trees Regression
# from sklearn import tree
# model = tree.DecisionTreeRegressor()

# Ensemble methods ()
# Either Bagging or Boosting
# Ensemble methods have a feature_importances_ attribute
# feature_importances_ gives a (n_features,) array which adds up to 1.0
# Bagging for averaging strong/complex models to reduce overfitting
# from sklearn.ensemble import BaggingRegressor
# bagging = BaggingRegressor(KNeighborsClassifier(),
#                            max_samples=0.5, max_features=0.5)
# from sklearn.ensemble import RandomForestRegressor
# bagging = RandomForestRegressor(n_estimators=10, max_depth=None,
#                                 min_samples_split=2, random_state=0,
#                                 max_features=None)
# from sklearn.ensemble import ExtraTreesRegressorhow
# bagging = ExtraTreesRegressor(...)

# Boosting for sequentially building weak models to reduce bias
# from sklearn.ensemble import AdaboostRegressor
# boosting = AdaBoostRegressor(n_estimators=100)
# from sklearn.ensemble import GradientBoostingRegressor
# boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#                                      max_depth=1, random_state=0, loss='ls')
# staged_predict() returns a generator that shows the predictions are each stage
# boosting.set_params(n_estimators=200, warm_start=True) allows addition of extra trees
# refit after setting



# 1.11.4.6 Regularization




# NN Models

# Create function for all similar methods

# Graph score for different parameters





'Dealing with null values: Remove, average, null-class, predict'

# Too much preprocessing done to data before ml_model step
# Next time, use more sklearn



'Setup Pipeline: indexers, encoders, assembler, label_indexer, pipe'

# Use pipeline, research what can be done
# Eg. preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100)

# Step by step process in the pipeline

# from sklearn.pipeline import make_pipeline
# pipeline = make_pipeline(step1, step2, ...)






# To-do: Drop features with null values
#X_train_original.isnull().sum()
#X_non_nulls = X_train_original.dropna(axis = 1)
#X_non_nulls.nunique().sort_values(ascending = True)

# To-do: Encode categorical values with numeric and then with binary
#X_selected = X_non_nulls.loc[:, X_non_nulls.nunique().sort_values()\
#                             < 50]
#cat_cols = list(X_selected.select_dtypes(['object']).columns.values)
#X_categorical = X_selected[cat_cols]. \
#                  apply(lambda x: x.astype('category').cat.codes)
#X_train_selected = X_train_numerical.join(X_categorical)
#clf = DecisionTreeClassifier()
#cv_score = cross_val_score(clf, 
#                            X_train_selected, y_train_original,
#                            scoring = 'accuracy',
#                            cv = 3,
#                            n_jobs = -1,
#                            verbose = 1)
#cv_score
# Testing on the test data
#clf.fit(X_train_selected, y_train_original)
#X_test_non_nulls = X_test_original.dropna(axis = 1)
#X_test_selected = X_test_non_nulls.loc[:, \
#                      X_test_non_nulls.nunique().sort_values() < 50] 
#cat_cols = list(X_test_selected.select_dtypes(['object']). \
#              columns.values)
#X_test_categorical = X_test_selected[cat_cols]. \
#                        apply(lambda x: \ 
#                                  x.astype('category').cat.codes)
#X_test_selected = X_test_numerical.join(X_test_categorical)
#y_pred = clf.predict(X_test_selected)
#y_pred = pd.DataFrame(data = y_pred, 
#                      index = X_test_selected.index.values,
#                      columns = ['status_group'])

# To-do: Regression forest (Random forest)
#X_train, X_test, y_train, y_test = train_test_split(
#    X_train_selected, y_train_original, test_size=0.2)
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
# Grid Search to find the best Random Forest Classifier
#param_grid = {
#    'n_estimators': [10, 20, 30],
#    'max_depth': [6, 10, 20, 30]
#}
#gridsearch = GridSearchCV(RandomForestClassifier(n_jobs = -1), 
#                          param_grid=param_grid, 
#                          scoring='accuracy', cv=3, 
#                          return_train_score=True, verbose=10)
#gridsearch.fit(X_train, y_train)
# Gets a list of parameters to use
#RandomForestClassifier().get_params().keys()
# Ranks parameters based on best score
#pd.DataFrame(gridsearch.cv_results_).sort_values( \
#                                         by='rank_test_score')
# Run on test data
#clf = RandomForestClassifier(max_depth = 20, 
#                             n_estimators = 30, 
#                             n_jobs = -1)
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)

# Running Machine Learning models (Training, CV, test)
# CV to allow tweaking of parameters to find the best ones
# To-do: Support Vector Machine SVM (Supervised Learning) No kernal first
# Adams optimisation?

# To-do: Neural Network

'''
Debugging a learning algorithm
-	Get more training examples
-	Try smaller set of features
-	Try additional features
-	Try polynomial features
-	Decrease lambda
-	Increase lambda
Best to do a diagnostic:
A test to gain insight on what is or isn’t working, and gain guidance on how best to improve performance.
Diagnostics can take time to implement but it can be a very good use of time.
'''

# Find features that have the most variance
# Polynomial features based on these features
# PCA to speed up Supervised Learning (only if model too big)

# To-do: Clustering (Unsupervised Learning)
# K-means