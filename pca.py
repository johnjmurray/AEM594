# -*- coding: utf-8 -*-
"""
PCR On 
@author: John Murray, The University of Alabama
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict
import scipy
from sklearn.preprocessing import scale


# PCA 

# load dataset into Pandas DataFrame
df = pd.read_csv('Data/Structures/archive_structure_data.csv', names=['Study Name','Craft Name','Non-Secondary Structures Mass','Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume'])
features = ['Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume']


pcColumns = ['pc1','pc2','pc3','pc4','pc5','pc6']
pcArray = list(range(1,7))

# Separating out the features from the Non-Secondary Structures Mass

x = df.loc[:, features].values


# Monte Carlo analysis
# Split data set randomly into training set and test set, run PCA on training set, transform test set using PCs, run linear regression on training set (Non-secondary Structural Mass vs. PCs), obtain regression coefficients, generate line of best fit, apply to transformed test data & generate predicted value for Non-secondary Structural Mass, find mean squared error between predicted and actual Non-Secondary Structural Mass for both training and testing data points. If sum of training + testing MSEs is less than the minimum sum yet found, record this split and retain it. If the sum is greater than the current "winner" split, do nothing. Loop for an arbitrary number of iterations. To cover all possible test/train split combinations would require 17! iterations. 

iter_n = 0
min_mseTrain = min_mseTest = 1e10

#dfTrain = min_dfTrain
#dfTest = min_dfTest
min_transformedTrainSet = True
while iter_n < 100000:
    dfTrain, dfTest = train_test_split(df,test_size=0.2) # split data into train/test
    trainData = dfTrain.loc[:,features].values
    testData = dfTest.loc[:,features].values
    
    # Run PCA on training data
    pca = PCA()
    transformedTrainSet = pca.fit_transform(trainData)
    dfTransformedTrainSet = pd.DataFrame(data = transformedTrainSet, columns = pcColumns)
    
    # Transform test set
    transformedTestSet = pca.transform(testData)
    dfTransformedTestSet = pd.DataFrame(data = transformedTestSet, columns = pcColumns)
    loadings = pd.DataFrame(pca.components_*np.sqrt(pca.explained_variance_), columns=pcColumns)
    
    # Run linear regression (Non-Secondary Structural Mass vs. training PCs)
    regr = linear_model.LinearRegression()
    y = dfTrain['Non-Secondary Structures Mass']
    regr.fit(transformedTrainSet,y)
    dfRegrCoef = pd.DataFrame(regr.coef_)
    
    # Generate list of predicted values for training data Non-Secondary Structural Mass, using the regression coefficients from above
    trainDataOnFitLine = []
    for ii in range(len(dfTransformedTrainSet)): # for each spacecraft element
        subElement = 0
        # multiply its value on the PC1 axis by the 1st regression coefficient, its value on the PC2 axis by the 2nd regression coefficient, etc., then sum and add the intercept value to generate predicted Non-Secondary Structural Mass
        for jj in range(len(dfRegrCoef)):
            subElement += dfRegrCoef.values[jj][0]*dfTransformedTrainSet.values[ii][jj]
        trainDataOnFitLine.append(subElement+regr.intercept_)
    # same for test data      
    testDataOnFitLine = []
    for ii in range(len(dfTransformedTestSet)):
        subElement = 0
        for jj in range(len(dfRegrCoef)):
            subElement += dfRegrCoef.values[jj][0]*dfTransformedTestSet.values[ii][jj]
        testDataOnFitLine.append(subElement+regr.intercept_)
    
    # subtract predicted mass values from recorded mass values, square the result, and find the mean of this squared error across the sets
    residsTrain = np.subtract(np.array(dfTrain['Non-Secondary Structures Mass']),trainDataOnFitLine)
    residsTest = np.subtract(np.array(dfTest['Non-Secondary Structures Mass']),testDataOnFitLine)
    mseTrain = np.average(residsTrain**2)
    mseTest = np.average(residsTest**2)
    
    # if the sum of the two MSEs is less than anything previously found, hold over all the information from this iteration.
    if mseTrain+mseTest < min_mseTrain+min_mseTest:
        min_mseTrain = mseTrain
        min_mseTest = mseTest
        min_dfTrain = dfTrain
        min_dfTest = dfTest
        minRegrCoef = dfRegrCoef
        min_iter_n = iter_n
        min_transformedTrainSet = transformedTrainSet
    iter_n+=1
    if iter_n%100 == 0:
        print('%s mseTrain = %f , mseTest = %f'%(str(iter_n),mseTrain,mseTest))
    
print('%s mseTrain = %f , mseTest = %f'%(str(min_iter_n),min_mseTrain,min_mseTest))
est = sm.OLS(min_dfTrain['Non-Secondary Structures Mass'], min_transformedTrainSet)
est2 = est.fit()
print(est2.summary())

### Pearson Correlation Coefficients between original variables
dfCorrs = pd.DataFrame(index=features,columns=features)
for ft1 in features:
    for ft2 in features:
        dfCorrs[ft1][ft2] = scipy.stats.pearsonr(scale(df[ft1]),scale(df[ft2]))[0]


### Plotting ###
#Scree plot (nice-to-have)
screeVec = pca.explained_variance_ratio_
screeAx = figure().gca()
plt.plot(pcArray,screeVec,'o--')
screeAx.set_ylabel('Explained Variance Fraction')
screeAx.set_xlabel('Component Number')
