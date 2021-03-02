#!/usr/bin/env python
# coding: utf-8

# In[13]:


import csv

with open('datasets/haberman.data.txt') as input_file:
    lines = input_file.readlines()
    newLines = []
    for line in lines:
        newLine = line.strip().split(',')
        newLines.append( newLine )

with open('datasets/output/haberman.csv', 'w') as test_file:
   
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )


# In[24]:


import pandas as pd

dataset = pd.read_csv ('datasets/output/haberman.csv', header = None)
dataset.columns = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']

dataset


# In[25]:


import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[27]:


dataset.plot()
plt.show()


# In[28]:


dataset.hist()
plt.show()


# In[29]:


array = dataset.values
X = array[:,:3]
Y = array[:,3]
validation_size = 0.30
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
test_size=validation_size, random_state=seed)


# In[30]:


# Test options and evaluation metric
num_folds = 20
seed = 10
scoring = 'accuracy'


# In[43]:


knn = KNeighborsClassifier()
kfold = model_selection.KFold(num_folds,  shuffle= True, random_state=seed)
cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold, scoring=scoring)

msg = "KNN: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)


# In[53]:


# Training using KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train,Y_train)


# In[54]:


predictions = classifier.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




