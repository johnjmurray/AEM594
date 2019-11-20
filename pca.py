# -*- coding: utf-8 -*-
"""
PCA On 
@author: John Murray, The University of Alabama
"""




from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from numpy import polynomial
from decimal import Decimal




def genTrainTest(dFrame, percentTest):
    # Inputs: pd dataframe of input data, specify what percentage of set should be reserved for testing
    # Outputs: 2 dataframes, 1 for training, one for testing
    splitPoint = np.int(len(dFrame)*(1-percentTest/100))
    trainFrame = dFrame[:splitPoint+1]
    testFrame = dFrame[splitPoint+1:]
    return trainFrame, testFrame


def regress(depVar,indVar,degree):
    # Inputs: dependent variable array, independent variable array, desired degree of fit equation
    # Outputs: regression coefficient and error tuple, y-values, x-values spanning input data
    regressionCoeffs = polynomial.polynomial.polyfit(depVar,indVar,degree,full=True)
    domain = np.arange(min(depVar)-0.1*abs(min(depVar)),max(depVar)+0.1*abs(max(depVar)))
    fitVals = np.zeros_like(domain)
    for ii in range(degree+1):
        yy = np.array(regressionCoeffs[0][ii]*domain**ii)
        fitVals += yy
    return regressionCoeffs, fitVals, domain


# PCA https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# load dataset into Pandas DataFrame
df = pd.read_csv('Data/Structures/archive_structure_data.csv', names=['Study Name','Craft Name','Non-Secondary Structures Mass','Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume'])
features = ['Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume']

pcColumns = ['pc1','pc2','pc3','pc4','pc5','pc6']
pcArray = list(range(1,7))

# Separating out the features

x = df.loc[:, features].values

dfTrain, dfTest = train_test_split(df,test_size=0.4) # split data into 60/40 train/test
dfTest, dfValidate = train_test_split(dfTest,test_size=0.5) # split test data 50/50 test/validate
trainData = dfTrain.loc[:,features].values
testData = dfTest.loc[:,features].values
validationData = dfValidate.loc[:,features].values

# Run PCA on training data
pca = PCA()
principalComponents = pca.fit_transform(trainData)
principalDf = pd.DataFrame(data = principalComponents, columns = pcColumns)

#Scree plot (nice-to-have)
screeVec = pca.explained_variance_
screeAx = figure().gca()
plt.plot(pcArray,screeVec)
screeAx.set_ylabel('Eigenvalues')
screeAx.set_xlabel('Component Number')
screeAx.set_title('Scree Plot')

# Multiple linear regression
r2list = []
r2listCv = []
mseList = []
mseListCv = []
for kk in range(1,7):        
    pca = PCA(n_components=kk)
    principalComponents = pca.fit_transform(trainData)
    principalDf = pd.DataFrame(data = principalComponents, columns = pcColumns[:kk])
    
    regr = linear_model.LinearRegression()
    y = dfTrain['Non-Secondary Structures Mass']
    regr.fit(principalComponents,y)
    y_c = regr.predict(principalComponents)
    y_cv = cross_val_predict(regr,principalComponents,y,cv=10)
    r2list.append(r2_score(y, y_c))
    r2listCv.append(r2_score(y, y_cv))
    # Calculate mean square error for calibration and cross validation
    mseList.append(mean_squared_error(y, y_c))
    mseListCv.append(mean_squared_error(y, y_cv))
    print(regr.intercept_,regr.coef_)

transformedTestSet = pca.transform(testData)
dfTransformedTestSet = pd.DataFrame(data = transformedTestSet, columns = pcColumns)

dfRegrCoef = np.transpose(pd.DataFrame(regr.coef_))


testDataOnFitLine = regr.intercept_ + np.transpose(np.dot(dfRegrCoef,np.transpose(dfTransformedTestSet)))
resids = list(dfTest['Non-Secondary Structures Mass']) - testDataOnFitLine
mse = np.average(resids**2)
print(mse)


r2Ax = figure().gca()
plt.plot(pcArray,r2list)
plt.plot(pcArray,r2listCv)
r2Ax.legend(['Calibration','Cross-Validation'])
r2Ax.set_xlabel('Component Number')
r2Ax.set_title('R2')

mseAx = figure().gca()
plt.plot(pcArray,mseList)
plt.plot(pcArray,mseListCv)
mseAx.legend(['Calibration','Cross-Validation'])
mseAx.set_xlabel('Component Number')
mseAx.set_title('MSE')

# BACKWARDS SELECTION??

'''
clrs = ['r','b','g']
for pcForRegression in pcColumns:
    ax = figure().gca()
    for regressDegree in range(1,4):
        
        # Regression fit
        regCoefs, fit, dmn = regress(principalDf[pcForRegression],dfTrain['Non-Secondary Structures Mass'],regressDegree)
        
        # Transform testing set from variable space into PC space
        transformedTestSet = pca.transform(testData)
        dfTransformedTestSet = pd.DataFrame(data = transformedTestSet, columns = pcColumns)
        
        
        # TODO: turn into fcn -- finds test data on regression fit line, finds residual of actual mass data to fit line
        # Evaluates goodness of fit
        testDataOnFitLine = np.zeros_like(dfTransformedTestSet[pcForRegression])
        for jj in range(regressDegree):
            testDataOnFitLine += np.array((regCoefs[0][jj]*dfTransformedTestSet[pcForRegression]**jj))
        resids = dfTest['Non-Secondary Structures Mass'] - testDataOnFitLine
        asq = np.average(resids**2)
        
        
        # Plot
        ax.set_xlabel(pcForRegression, fontsize = 15)
        ax.set_ylabel('Mass (kg)', fontsize = 15)
        plt.scatter(principalDf[pcForRegression],dfTrain['Non-Secondary Structures Mass'],c='k')
        plt.plot(dmn,fit,c=clrs[regressDegree-1],label='Degree %s Fit, AvSQ %2.E'%(regressDegree,Decimal(asq)))
        plt.scatter(dfTransformedTestSet[pcForRegression],dfTest['Non-Secondary Structures Mass'],c='y')
    plt.legend(loc='best')
    plt.show()
    ax.grid()


'''
