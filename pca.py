# -*- coding: utf-8 -*-
"""
PCA On 
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
while iter_n < 100000:
    dfTrain, dfTest = train_test_split(df,test_size=0.2) # split data into train/test
    trainData = dfTrain.loc[:,features].values
    testData = dfTest.loc[:,features].values
    
    # Run PCA on training data
    pca = PCA()
    principalComponents = pca.fit_transform(trainData)
    principalDf = pd.DataFrame(data = principalComponents, columns = pcColumns)
    
    transformedTestSet = pca.transform(testData)
    dfTransformedTestSet = pd.DataFrame(data = transformedTestSet, columns = pcColumns)
    comps = pd.DataFrame(pca.components_, columns=pcColumns)
    
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
    
    if mseTrain+mseTest < min_mseTrain+min_mseTest:
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
est = sm.OLS(min_dfTrain['Non-Secondary Structures Mass'], principalComponents)
est2 = est.fit()
print(est2.summary())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(list(principalDf['pc1']),list(principalDf['pc2']),list(dfTrain['Non-Secondary Structures Mass']))
ax.scatter(list(dfTransformedTestSet['pc1']),list(dfTransformedTestSet['pc2']),list(dfTest['Non-Secondary Structures Mass']))
pc1Domain = np.linspace(min(list(principalDf['pc1'])),max(list(principalDf['pc1'])),1000)
pc2Domain = np.linspace(min(list(principalDf['pc2'])),max(list(principalDf['pc2'])),1000)
fitLine = regr.intercept_+dfRegrCoef[0][0]*pc1Domain+dfRegrCoef[1][0]*pc2Domain
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


### histogram plots of original data

fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=False)
for pltCount in range(6):
    print(pltCount)
    print(axs.flat[pltCount])
    print(list(df[features[pltCount]]))
    axs.flat[pltCount].hist(x=np.array(df[features[pltCount]]), color='#0504aa',alpha=0.7, rwidth=0.85)
    axs.flat[pltCount].set_title(features[pltCount])
fig.tight_layout()
plt.show()


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
