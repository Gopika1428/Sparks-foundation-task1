#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION :GRADUATE ROTATIONAL INTERNSHIP PROGRAM

# # #Domain:Data Science and Bussiness Analytics

# TASK-1 Prediction using Supervised ML:Predict the percentage of an student based on the no. of study hours.

# In[3]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[4]:


#import dataset
data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


plt.scatter(data.Hours,data.Scores)
plt.title('Hours v/s Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')


# In[10]:


data.corr()


# In[16]:


x = data.iloc[: , :-1].values
y = data.iloc[: ,1 ].values


# In[17]:


x


# In[18]:


y


# spliting Training and Test sets

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# Training model

# In[24]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[37]:


line = regressor.coef_*x+regressor.fit_intercept
plt.scatter(x,y, c=y,cmap='Set1')
plt.plot(x,line)
plt.colorbar()
plt.show()


# In[28]:


print(x_test)


# In[29]:


print(y_test)


# In[30]:


print(x_train)


# In[31]:


print(y_train)


# Predicting model

# In[33]:


y_pred = regressor.predict(x_test)
y_pred


# checking r2 score and mean absolute error

# In[43]:


from sklearn import metrics
MAE=metrics.mean_absolute_error(y_pred,y_test)
MAE


# Comparing actual and predicted target labels

# In[38]:


df = pd.DataFrame({'Actual':y_test,'predicted':y_pred})
df


# Checking the accuracy of model

# In[44]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# # Q:What will be the predicted score if a student studies for 9.25hr/day?

# In[46]:


soln_pred = regressor.predict([[9.25]])
soln_pred


# ## Predicted score 93.69%

# In[ ]:




