#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import read_excel


# In[2]:


# load python using pandas
filename = 'datasets/Stats.xlsx'


# In[3]:


data = read_excel(filename)


# In[4]:


data


# In[5]:


data_filtered = data.iloc[:, 1:]
data_filtered.mean()


# In[6]:


data_filtered.median()


# In[7]:


data_filtered.mode()


# In[8]:


import numpy as np


# In[9]:


#calculate variation and standard deviation using numpy
exp_data = data['YearsOfExp']
exp_var = np.var(exp_data)
print("Varance of Years of Expereince")
print(exp_var)
print("Standard Deviation of Years of Expereince")
exp_sd = np.std(exp_data)
print(exp_sd)


# In[10]:


sal_data = data['Salary in Rs.']
sal_var = np.var(sal_data)
print("Varance of Salary")
print(sal_var)
print("Standard Deviation of Salary")
sal_sd = np.std(sal_data)
print(sal_sd)


# In[ ]:




