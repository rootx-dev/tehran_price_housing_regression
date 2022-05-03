#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


get_ipython().run_line_magic('matplotlib', 'inline')


# # bring data

# In[2]:


df = pd.read_csv("Houses.csv" , dtype={'Area': int, 'Address': str})
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df['Price(USD)'] = np.array(df['Price(USD)'],dtype='int64')
df['Address'] = np.array(df['Address']).astype(str)


# In[5]:


df.info()


# In[ ]:





# In[6]:


print(df['Area'])
print(df['Room'])


# In[7]:


plt.scatter(df.Area, df.Price,  color='blue')
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


# In[8]:


plt.scatter(df.Room, df.Price,  color='blue')
plt.xlabel("Room")
plt.ylabel("Price")
plt.show()


# In[9]:


plt.scatter(df.Parking, df.Price,  color='blue')
plt.xlabel("Parking")
plt.ylabel("Price")
plt.show()


# In[10]:


viz = df[['Area', 'Room', 'Price', 'Price(USD)']]
viz.hist()
plt.show()


# In[12]:


def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df


# In[18]:


cdf = Encoder(df)


# In[19]:


print(df)


# In[21]:


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# In[ ]:





# In[22]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['Area','Room','Parking','Warehouse','Elevator','Address']])
y = np.asanyarray(train[['Price']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[ ]:





# In[ ]:




