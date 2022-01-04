#!/usr/bin/env python
# coding: utf-8

# # CSSM502 Advance Data Analysis in Python - Homework 3
# ### İzel Yazıcı - Student ID: 0077549
# The purpose of this homework is to review and practice fundamental machine learning concepts.
# I will present the applications of different classifiers used for binary classification

# To ignore warnings
import warnings


## For data prepocessing 
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

# For ML models and Feature Selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# For Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV, KFold


# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# Reading cses4_cut.csv and splitting to train and test set.

df = pd.read_csv('cses4_cut.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=97)

print(df.head())


# ## Classifiers without reduction
# #### Logistic Regression

cross_validation = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression_accuracy = cross_val_score(logistic_regression, X, y, cv=cross_validation).mean()


from sklearn.linear_model import TweedieRegressor
reg = TweedieRegressor(power=1, alpha=0.5, link='log')



#### 1-Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy = cross_val_score(decision_tree, X, y, cv=cross_validation).mean()

#### 2-Support Vector Machine
SVM = SVC(probability=True)
SVM_accuracy = cross_val_score(SVM, X, y, cv=cross_validation).mean()

#### 3- Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy = cross_val_score(LDA, X, y, cv=cross_validation).mean()

#### 4- Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy = cross_val_score(QDA, X, y, cv=cross_validation).mean()

#### 5- Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy = cross_val_score(random_forest, X, y, cv=cross_validation).mean()

#### 6- K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy = cross_val_score(KNN, X, y, cv=cross_validation).mean()

#### 7- Naive Bayes
bayes = GaussianNB()
BAYES_accuracy = cross_val_score(bayes, X, y, cv=cross_validation).mean()

###

pd.options.display.float_format = '{:,.2f}%'.format
accuracies1 = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis',
              'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Accuracy': [100 * logistic_regression_accuracy, 100 * DT_accuracy, 100 * SVM_accuracy, 100 * LDA_accuracy,
                 100 * QDA_accuracy, 100 * RF_accuracy, 100 * KNN_accuracy, 100 * BAYES_accuracy],
}, columns=['Model', 'Accuracy'])

accuracies1.sort_values(by='Accuracy', ascending=False)


# ## Feature Selection and Dimensionality Reduction

#Select features according to the k the highest scores

test = SelectKBest(score_func=chi2, k='all')
fit = test.fit(X, y)
kscores = fit.scores_
X_new = test.fit_transform(X, y)

# Features in descending order by score
dicts = {}
dicts = dict(zip(df.columns, kscores))
sort_dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)

# 10 features with the highest score

sort_dicts[:10]

# 10 features with the highest score column names
X_new = df[['D2011', 'D2021', 'D2022', 'D2023', 'D2026', 'D2027', 'D2028', 'D2029', 'D2030']]

# new table after feature selection
print(X_new)

# data distribution of new table

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in X_new:
    if plotnumber <= 10:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(X_new[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()


# since the data distribution of new table is not Gaussian I did pre-processing and transform it in Gaussian form
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_new_trans = quantile_transformer.fit_transform(X_new)


#After preprocessing,now data is in Gaussian Form

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in range(X_new_trans.shape[1]):
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(X_new_trans[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()


# ## Classifiers with dimensionality-reduction and pre-processing

#### Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression_accuracy = cross_val_score(logistic_regression, X_new_trans, y, cv=cross_validation).mean()

#### Decision Tree
decision_tree = DecisionTreeClassifier()
DT_accuracy = cross_val_score(decision_tree, X_new_trans, y, cv=cross_validation).mean()

#### Support Vector Machine
SVM = SVC(probability=True)
SVM_accuracy = cross_val_score(SVM, X_new_trans, y, cv=cross_validation).mean()

#### Linear Discriminant Analysis
LDA = LinearDiscriminantAnalysis()
LDA_accuracy = cross_val_score(LDA, X_new_trans, y, cv=cross_validation).mean()

#### Quadratic Discriminant Analysis
QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy = cross_val_score(QDA, X_new_trans, y, cv=cross_validation).mean()

#### Random Forest Classifier
random_forest = RandomForestClassifier()
RF_accuracy = cross_val_score(random_forest, X_new_trans, y, cv=cross_validation).mean()

#### K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN_accuracy = cross_val_score(KNN, X_new_trans, y, cv=cross_validation).mean()

#### Naive Bayes
bayes = GaussianNB()
BAYES_accuracy = cross_val_score(bayes, X_new_trans, y, cv=cross_validation).mean()

pd.options.display.float_format = '{:,.2f}%'.format
accuracies2 = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis',
              'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Accuracy': [100 * logistic_regression_accuracy, 100 * DT_accuracy, 100 * SVM_accuracy, 100 * LDA_accuracy,
                 100 * QDA_accuracy, 100 * RF_accuracy, 100 * KNN_accuracy, 100 * BAYES_accuracy],
}, columns=['Model', 'Accuracy'])

accuracies2.sort_values(by='Accuracy', ascending=False)


# ## Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

seed=42
grid={"C":np.logspace(-3,3,7)
      ,"penalty":["l1","l2"] # l1 for lasso regression, l2 for ridge  regression
      ,"verbose":[1,5,10]}

logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10) # we can also define n_jobs=10 parallel processing
logreg_cv.fit(X_train,y_train)
best_parameters=logreg_cv.best_params_

clf = LogisticRegression(C=logreg_cv.best_params_['C']
                         ,penalty=logreg_cv.best_params_['penalty']
                         # ,verbose=logreg_cv.best_params_['verbose']
                         ,random_state=42).fit(X, y)

# Setting regularization parameter: The alpha parameter controls the degree of sparsity of the estimated coefficients

pred = clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_test, y_test)


# After I have tried GridSearchCV, I realized that doesnt gives us better accuracy if I compare with some hyper parameter tuning codes that I wrote from scracth as a bunch of parameter loop, thats why I used them for this part. 
# GridSearch for SVM

# grid=[{'kernel': ['rbf'],
#        'gamma': [1e-3, 1e-4],
#        'C': [1, 10, 100]}
#       # ,{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
#       ]
#
# clf_cv = GridSearchCV(svm.SVC(), grid)
# clf_cv.fit(X_train,y_train)
# best_parameters=clf_cv.bestparams
#
# clf_svm = svm.SVC(kernel=clf_cv.bestparams['kernel']
#                          ,gamma=clf_cv.bestparams['gamma']
#                          ,C=clf_cv.bestparams['C']
#                          ,random_state=42).fit(X_train, y_train)
#
# pred_svm = clf_svm.predict(X_test)


# GridSearch for Random Forest

# grid = {"n_estimators":[100],
#         "criterion":["gini", "entropy"],
#         "max_depth":[None],
#         "min_samples_split":[2],
#         "min_samples_leaf":[1],
#         "min_weight_fraction_leaf":[0.0],
#         "max_features":["auto", "sqrt", "log2"],
#         "max_leaf_nodes":[None],
#         "min_impurity_decrease":[0.0],
#         "bootstrap":[True],
#         "oob_score":[False],
#         "n_jobs":[None],
#         "random_state":[None],
#         "verbose":[0],
#         "warm_start":[False],
#         "class_weight":["balanced", "balanced_subsample"],
#         "ccp_alpha":[0.0],
#         "max_samples":[None]}
#
# clf_rf = RandomForestClassifier()
#
# gb_cv=GridSearchCV(clf_rf,grid,cv=10)
#
# gb_cv.fit(X_train,y_train)
#
# # asagidaki parametreler gridsearche konulan parametrelere gore degiskenlik gosterir.
# clf_rf = RandomForestClassifier(n_estimators=100,
#                         criterion='gini', #criterion{“gini”, “entropy”}
#                         max_depth=None,
#                         min_samples_split=2,
#                         min_samples_leaf=1,
#                         min_weight_fraction_leaf=0.0,
#                         max_features='auto', #max_features{“auto”, “sqrt”, “log2”}
#                         max_leaf_nodes=None,
#                         min_impurity_decrease=0.0,
#                         min_impurity_split=None,
#                         bootstrap=True,
#                         oob_score=False,
#                         n_jobs=None,
#                         random_state=None,
#                         verbose=0,
#                         warm_start=False,
#                         class_weight=None, #class_weight{“balanced”, “balanced_subsample”}
#                         ccp_alpha=0.0,
#                         max_samples=None)


### Top 5 classifier and tried to find the best hyperparameter

#### Random Forest Classifier

best_score = 0
n_estimators = [100, 200, 500, 1000]
criterions = ['gini', 'entropy']
for i in n_estimators:
    for k in criterions:
        random_forest = RandomForestClassifier(n_estimators=i, criterion=k)
        RF_accuracy = cross_val_score(random_forest, X_new_trans, y, cv=cross_validation).mean()
        if RF_accuracy > best_score:
            best_score = RF_accuracy
            best_est = i
            best_cri = k
RF_accuracy = best_score
print("Best score is:", best_score, "with estimator:", best_est, "criterion:", best_cri)

#### Linear Discriminant Analysis

best_score = 0
solver = ['svd', 'lsqr', 'eigen']
for i in solver:
    LDA = LinearDiscriminantAnalysis(solver=i)
    LDA_accuracy = cross_val_score(LDA, X_new_trans, y, cv=cross_validation).mean()
    if LDA_accuracy > best_score:
        best_score = LDA_accuracy
        best_solver = i
LDA_accuracy = best_score
print("Best score is:", best_score, "with solver:", best_solver)

#### Logistic Regression

best_score = 0
penalty = ['l1', 'l2', 'elasticnet', 'none']
for i in penalty:
    logistic_regression = LogisticRegression(penalty=i)
    logistic_regression_accuracy = cross_val_score(logistic_regression, X_new_trans, y, cv=cross_validation).mean()
    if logistic_regression_accuracy > best_score:
        best_score = logistic_regression_accuracy
        best_p = i
logistic_regression_accuracy = best_score
print("Best score is:", best_score, "with penalty", best_p)

#### K-Nearest Neighbors

best_score = 0
for i in range(2, 10):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN_accuracy = cross_val_score(KNN, X_new_trans, y, cv=cross_validation).mean()
    if KNN_accuracy > best_score:
        best_score = KNN_accuracy
        best_n = i
KNN_accuracy = best_score
print("Best score is:", best_score, "with number of neighbors:", best_n)

#### Support Vector Machine

best_score = 0
clist = [0.1, 1, 2, 5]
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed2']
for i in clist:
    for k in kernel:
        SVM = SVC(C=i, kernel=k)
        SVM_accuracy = cross_val_score(SVM, X_new_trans, y, cv=cross_validation).mean()
        if SVM_accuracy > best_score:
            best_score = SVM_accuracy
            best_c = i
            best_k = k
SVM_accuracy = best_score
print("Best score is:", best_score, "with c:", best_c, "kernel:", k)

pd.options.display.float_format = '{:,.2f}%'.format
accuracies3 = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Random Forest',
              'K-Nearest Neighbors', ],
    'Accuracy': [100 * logistic_regression_accuracy, 100 * SVM_accuracy, 100 * LDA_accuracy, 100 * RF_accuracy,
                 100 * KNN_accuracy],
}, columns=['Model', 'Accuracy'])

accuracies3.sort_values(by='Accuracy', ascending=False)


# # FINAL RESULTS

print("Classifiers without reduction:")
print(accuracies1.sort_values(by='Accuracy', ascending=False))
print("Classifiers with dimensionality-reduction and pre-processing:")
print(accuracies2.sort_values(by='Accuracy', ascending=False))
print("After optimizing the model and its hyperparameters:")
print(accuracies3.sort_values(by='Accuracy', ascending=False))




