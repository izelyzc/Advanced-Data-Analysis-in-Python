# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:41:06 2021

@author: Izel Yazici 
"""

import numpy as np
import pandas as pd
import kaggle as kg
import scipy.stats as st


### Getting CalCOFI dataset by using kaggle API
kg.api.authenticate()
kg.api.dataset_download_file('sohier/calcofi', file_name='bottle.csv',  path='data/')

#read data from csv
df=pd.read_csv('data/bottle.csv.zip',nrows=1000)

#Write first 1000 rows of the data to csv file
df.to_csv("calcofi_data.csv")

def LinearReg(df):
    Beta, error, t1, t2 = None, None, None, None

    # Handling NaN value
    df = df.dropna()
    X=df[['T_degC','Depthm']].to_numpy()
    y=df['Salnty'].to_numpy()
     

    ## Adding 1 to the X matrix for the intercept
    X_one = np.concatenate((np.ones((len(X), 1), dtype=int), X), axis=1)

    ## Beta
    Beta = np.linalg.solve(np.dot(X_one.T, X_one), np.dot(X_one.T, y))


    # Y_hat
    n = X.shape[1] ##number of obs.
    h = np.ones((X.shape[0], 1))
    theta = Beta.reshape(1, n)
    for i in range(0, X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    y_hat = h.reshape(X.shape[0])

    # Residuals
    e = np.subtract(y, y_hat)
    ss_numerator = np.dot(e.T, e)
    sample_no = X.shape[0]
    denom = sample_no - n

    # Variance of Beta
    sigma_squared = np.true_divide(ss_numerator, denom)
    inverse = np.linalg.inv(np.dot(X.T, X))
    variance_B = np.dot(sigma_squared, inverse)
    var_B = np.diag(variance_B)

    # Standard error of Beta
    error = np.sqrt(var_B)

    # T-statistics
    t = st.stats.t.ppf(0.95, denom)
    t_part = t * error

    # Credible intervals (t2 upper and t1 lower)
    upper = np.add(theta, t_part)
    t2 = upper.reshape(-1, )
    lower = np.subtract(theta, t_part)
    t1 = lower.reshape(-1, )
    return Beta, error, t1, t2, X, y

#a,b,c,d,e,f=LinearReg(df)
