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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import numpy as np
from numpy import polynomial
from decimal import Decimal
import statsmodels.api as sm
from scipy import stats


'''

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
'''

# PCA https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# load dataset into Pandas DataFrame
df = pd.read_csv('Data/Structures/archive_structure_data.csv', names=['Study Name','Craft Name','Non-Secondary Structures Mass','Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume'])
features = ['Fuel/Oxidizer Tanks Mass','Total Dry Mass','Total Wet Mass','Prop Mass','Surface Area','Volume']

pcColumns = ['pc1','pc2','pc3','pc4','pc5','pc6']
pcArray = list(range(1,7))

# Separating out the features

x = df.loc[:, features].values

iter_n = 0
min_mseTrain = min_mseTest = 1e10

dfTrain = min_dfTrain
dfTest = min_dfTest
while iter_n < 100:
    #dfTrain, dfTest = train_test_split(df,test_size=0.2) # split data into train/test
    #dfTest, dfValidate = train_test_split(dfTest,test_size=0.5) # split test data 50/50 test/validate
    trainData = dfTrain.loc[:,features].values
    testData = dfTest.loc[:,features].values
    #validationData = dfValidate.loc[:,features].values
    
    # Run PCA on training data
    pca = PCA()
    principalComponents = pca.fit_transform(trainData)
    principalDf = pd.DataFrame(data = principalComponents, columns = pcColumns)
    
    transformedTestSet = pca.transform(testData)
    dfTransformedTestSet = pd.DataFrame(data = transformedTestSet, columns = pcColumns)
    
    
    regr = linear_model.LinearRegression()
    y = dfTrain['Non-Secondary Structures Mass']
    regr.fit(principalComponents,y)
    dfRegrCoef = np.transpose(pd.DataFrame(regr.coef_))
    
    trainDataOnFitLine = regr.intercept_ + np.transpose(np.dot(dfRegrCoef,np.transpose(principalDf)))
    testDataOnFitLine = regr.intercept_ + np.transpose(np.dot(dfRegrCoef,np.transpose(dfTransformedTestSet)))
    residsTrain = list(dfTrain['Non-Secondary Structures Mass']) - trainDataOnFitLine
    residsTest = list(dfTest['Non-Secondary Structures Mass']) - testDataOnFitLine
    mseTrain = np.average(residsTrain**2)
    mseTest = np.average(residsTest**2)
    
    if mseTrain < min_mseTrain and mseTest < min_mseTest:
        min_mseTrain = mseTrain
        min_mseTest = mseTest
        min_dfTrain = dfTrain
        min_dfTest = dfTest
        minRegrCoef = dfRegrCoef
        min_iter_n = iter_n
    iter_n+=1
    if iter_n%100 == 0:
        print('%s mseTrain = %f , mseTest = %f'%(str(iter_n),mseTrain,mseTest))
    
print('%s mseTrain = %f , mseTest = %f'%(str(min_iter_n),min_mseTrain,min_mseTest))
est = sm.OLS(y, principalComponents)
est2 = est.fit()
print(est2.summary())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(list(principalDf['pc1']),list(principalDf['pc2']),list(dfTrain['Non-Secondary Structures Mass']))
ax.scatter(list(dfTransformedTestSet['pc1']),list(dfTransformedTestSet['pc2']),list(dfTest['Non-Secondary Structures Mass']))
pc1Domain = np.linspace(min(list(principalDf['pc1'])),max(list(principalDf['pc1'])),1000)
pc2Domain = np.linspace(min(list(principalDf['pc2'])),max(list(principalDf['pc2'])),1000)
fitLine = dfRegrCoef[0]*pc1Domain+dfRegrCoef[1]*pc2Domain

ax.scatter(pc1Domain,pc2Domain,fitLine)
'''
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
'''



### Plotting ###
'''
#Scree plot (nice-to-have)
screeVec = pca.explained_variance_ratio_
screeAx = figure().gca()
plt.plot(pcArray,screeVec)
screeAx.set_ylabel('Explained Variance Fraction')
screeAx.set_xlabel('Component Number')
screeAx.set_title('Scree Plot')
'''
'''
# r-squared & mean squared error plots

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
'''
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
