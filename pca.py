# -*- coding: utf-8 -*-
"""
Principal component regression for structural mass predictions of spacecraft
@author: John Murray, The University of Alabama
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import statsmodels.api as sm
import scipy
from sklearn.preprocessing import scale


"""
PCA 
load dataset into Pandas DataFrame
"""
df = pd.read_csv('data/archive_structure_data_cleaned.csv', names=['Study Name','Craft Name','Non-Secondary Structures Mass','Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume'])
features = ['Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume']


pc_columns = ['pc1','pc2','pc3','pc4','pc5','pc6']
pc_array = list(range(1,7))

"""Separating out the features from the Non-Secondary Structures Mass"""

x = df.loc[:, features].values


"""
Monte Carlo analysis
Split data set randomly into training set and test set, run PCA on training set, transform test set using PCs, run linear regression on training set (Non-secondary Structural Mass vs. PCs), obtain regression coefficients, generate line of best fit, apply to transformed test data & generate predicted value for Non-secondary Structural Mass, find mean squared error between predicted and actual Non-Secondary Structural Mass for both training and testing data points. If sum of training + testing MSEs is less than the minimum sum yet found, record this split and retain it. If the sum is greater than the current "winner" split, do nothing. Loop for an arbitrary number of iterations. To cover all possible test/train split combinations would require 17! iterations.
"""

iter_n = 0
min_mse_train = min_mse_test = 1e10

#df_train = min_df_train
#df_test = min_df_test
min_transformed_train_set = True

while iter_n < 100000:
    df_train, df_test = train_test_split(df,test_size=0.2) # split data into train/test
    train_data = df_train.loc[:,features].values
    test_data = df_test.loc[:,features].values
    
    """Run PCA on training data"""
    pca = PCA()
    transformed_train_set = pca.fit_transform(train_data)
    df_transformed_train_set = pd.DataFrame(data = transformed_train_set, columns = pc_columns)
    
    """Transform test set"""
    transformed_test_set = pca.transform(test_data)
    df_transformed_test_set = pd.DataFrame(data = transformed_test_set, columns = pc_columns)
    loadings = pd.DataFrame(pca.components_*np.sqrt(pca.explained_variance_), columns=pc_columns)
    
    """Run linear regression (Non-Secondary Structural Mass vs. training PCs)"""
    regr = linear_model.LinearRegression()
    y = df_train['Non-Secondary Structures Mass']
    regr.fit(transformed_train_set,y)
    df_regr_coef = pd.DataFrame(regr.coef_)
    
    """Generate list of predicted values for training data Non-Secondary Structural Mass, using the regression coefficients from above"""
    train_data_on_fit_line = []
    for ii in range(len(df_transformed_train_set)): # for each spacecraft element
        sub_element = 0
        """Multiply its value on the PC1 axis by the 1st regression coefficient, its value on the PC2 axis by the 2nd regression coefficient, etc., then sum and add the intercept value to generate predicted Non-Secondary Structural Mass"""
        for jj in range(len(df_regr_coef)):
            sub_element += df_regr_coef.values[jj][0]*df_transformed_train_set.values[ii][jj]
        train_data_on_fit_line.append(sub_element+regr.intercept_)
    
    """Same for test data"""
    test_data_on_fit_line = []
    for ii in range(len(df_transformed_test_set)):
        sub_element = 0
        for jj in range(len(df_regr_coef)):
            sub_element += df_regr_coef.values[jj][0]*df_transformed_test_set.values[ii][jj]
        test_data_on_fit_line.append(sub_element+regr.intercept_)
    
    """Subtract predicted mass values from recorded mass values, square the result, and find the mean of this squared error across the sets"""
    resids_train = np.subtract(np.array(df_train['Non-Secondary Structures Mass']),train_data_on_fit_line)
    resids_test = np.subtract(np.array(df_test['Non-Secondary Structures Mass']),test_data_on_fit_line)
    mse_train = np.average(resids_train**2)
    mse_test = np.average(resids_test**2)
    
    """If the sum of the two MSEs is less than anything previously found, hold over all the information from this iteration"""
    if mse_train+mse_test < min_mse_train+min_mse_test:
        min_mse_train = mse_train
        min_mse_test = mse_test
        min_df_train = df_train
        min_df_test = df_test
        minRegrCoef = df_regr_coef
        min_iter_n = iter_n
        min_transformed_train_set = transformed_train_set
    iter_n+=1
    if iter_n%100 == 0:
        print('%s mse_train = %f , mse_test = %f'%(str(iter_n),mse_train,mse_test))
    
print('%s mse_train = %f , mse_test = %f'%(str(min_iter_n),min_mse_train,min_mse_test))
est = sm.OLS(min_df_train['Non-Secondary Structures Mass'], min_transformed_train_set)
est2 = est.fit()
print(est2.summary())

"""Pearson Correlation Coefficients between original variables"""
dfCorrs = pd.DataFrame(index=features,columns=features)
for ft1 in features:
    for ft2 in features:
        dfCorrs[ft1][ft2] = scipy.stats.pearsonr(scale(df[ft1]),scale(df[ft2]))[0]

"""
Plotting
Scree plot (nice-to-have)
"""
scree_vec = pca.explained_variance_ratio_
scree_ax = figure().gca()
plt.plot(pc_array,scree_vec,'o--')
scree_ax.set_ylabel('Explained Variance Fraction')
scree_ax.set_xlabel('Component Number')
