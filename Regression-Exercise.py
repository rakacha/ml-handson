#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

with open('datasets/auto-mpg.data.txt') as input_file:
    lines = input_file.readlines()
    newLines = []
    for line in lines:
        newLine = line.strip().split('\t')
        valSp =  newLine[0].split()
       
        nameSp = newLine[1].replace('"', '')
        
        valSp.append(nameSp)
        newLines.append( valSp )

with open('datasets/output/auto-mpg.csv', 'w') as test_file:
   
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )


# In[2]:


import pandas as pd

read_file = pd.read_csv ('datasets/output/auto-mpg.csv', header = None)
read_file.columns = ['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

read_file


# In[3]:


data = read_file[read_file.horsepower != '?']
data.horsepower = data.horsepower.astype('float')

data.dtypes


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


def get_data(columns):
    X = pd.DataFrame(data[columns].copy())
    return X
    


# In[6]:


X = get_data('horsepower')
y = data['mpg'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=324)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_predicted = regressor.predict(X_test)


# In[7]:


# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Intercepts: \n', regressor.intercept_)
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_predicted))
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
print('Root mean squared error: %.2f'
      % rmse)


# In[8]:


# Visualising the training results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Horsepower vs MPG(Training set)')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()


# In[9]:


# Visualising the training results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, regressor.predict(X_test), color = 'red')
plt.title('Horsepower vs MPG(Test set)')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()


# In[10]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
X = get_data(factors)
X = StandardScaler().fit_transform(X)
y = data['mpg'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=324)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_predicted = regressor.predict(X_test)


# In[11]:


# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Intercepts: \n', regressor.intercept_)
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_predicted))
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
print('Root mean squared error using simple linear regression model: %.2f'
      % rmse)


# In[12]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
scaled_data = get_data(factors)
X = StandardScaler().fit_transform(scaled_data)
# X = scaled_data.iloc[:,:]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = get_data(['mpg']).iloc[:,:].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,8])


# In[13]:


def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


# In[14]:


def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost


# In[15]:


theta = np.zeros([1,8])
alpha = 0.01
iters = 1000
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)


# In[16]:


finalCost = computeCost(X,y,g)
print(finalCost)


# In[17]:


fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')
plt.show()


# In[18]:


# calculate optimum theta value
theta = np.zeros([1,8])
alpha = 0.01
iters = 350
g,cost = gradientDescent(X,y,theta,iters,alpha)
print('Optmum gradiant')
print(g)


# In[19]:


# calculate optimum cost function
finalCost = computeCost(X,y,g)
print('Optimum cost: %.2f'
      % finalCost)


# In[20]:


y_pred = X@g.T


# In[21]:


rmse = sqrt(mean_squared_error(y_true=y,y_pred=y_pred))
print('Root mean squared error using manual gradient descent regesssion model: %.2f'
      % rmse)


# In[22]:


from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[23]:


gb_regressor = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01)
gb_regressor.fit(X_train,y_train)


# In[24]:


y_predicted_gbr = gb_regressor.predict(X_test)
rmse_bgr = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr))
print('Root mean squared error GB: %.2f' % rmse_bgr)

fi= pd.Series(gb_regressor.feature_importances_,index=factors)
fi.plot.barh()


# In[25]:


cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)
params = {'n_estimators' : list(range(40,60)),
         'max_depth' : list(range(1,10)),
         'learning_rate' : [0.1, 0.2, 0.3] }
grid = GridSearchCV(gb_regressor, params,cv=cv_sets,n_jobs=4)

grid = grid.fit(X_train, y_train) 
grid.best_estimator_


# In[26]:


gb_regressor_t = grid.best_estimator_
gb_regressor_t.fit(X_train,y_train)
y_predicted_gbr_t = gb_regressor_t.predict(X_test)
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))
print('Root mean squared error using gradiant boost regressor model: %.2f'
      % rmse)


# In[27]:


sgd_regressor = SGDRegressor()


# In[28]:


# Grid search - this will take about 1 minute.
cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)
param_grid = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
}
clf = GridSearchCV(sgd_regressor, param_grid,cv=cv_sets,n_jobs=4)
clf.fit(X_train, y_train)
print("Best score: " + str(clf.best_score_))


# In[29]:



clf.best_estimator_


# In[30]:


sgd_regressor_t = clf.best_estimator_
sgd_regressor_t.fit(X_train,y_train)
y_predicted_gbr_t = sgd_regressor_t.predict(X_test)
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))
print('Root mean squared error using SGD regressor model: %.2f'
      % rmse)


# In[ ]:




