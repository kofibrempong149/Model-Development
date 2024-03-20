#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv("C:/Users/HP/Desktop/Life Expectancy Data.csv")


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.corr()


# In[7]:


correlation_metrics = df.corr()


# In[9]:


target_correlation = correlation_metrics["GDP"].sort_values(ascending=False)
print(target_correlation )


# In[10]:


df_G = df[df["Country"] == 'Ghana']


# In[11]:


df_G


# In[12]:


df_G.dropna(inplace=True)


# In[13]:


df_G


# In[15]:


x = df_G[["percentage expenditure", "Life expectancy ", "Income composition of resources", "Schooling"]]
y = df_G[["GDP"]]


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(x_train, y_train)


# In[20]:


prediction = model.predict(x_test)
print(prediction)


# In[21]:


mse = mean_squared_error(y_test, prediction)
print(mse)


# In[23]:


lm = LinearRegression()
x = df_G[["percentage expenditure", "Life expectancy ", "Income composition of resources", "Schooling"]]
y = df_G[["GDP"]]
lm.fit(x, y)
lm.score(x, y)


# In[ ]:




