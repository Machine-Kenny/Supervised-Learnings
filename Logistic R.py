#!/usr/bin/env python
# coding: utf-8

# ### Sigmoid Function acts as an activation function in machine learning which is used to add non-linearity in a machine learning model, in simple words it decides which value to pass as output and what not to pass, there are mainly 7 types of Activation Functions which are used in machine learning and deep learning

# In[1]:


import pandas as pd
data = pd.read_csv("C:\Python_Files\insurance_logistic.csv")


# In[2]:


data.head()


# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.scatter(data.age, data.insurance, marker='+', color='blue')


# ##splitting the data into training and test - we check the shape of our dataset first and yhen we import sklearn to do the analysis

# In[5]:


data.shape


# In[6]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(data[['age']], data.insurance, test_size=0.2)


# In[29]:


X_test


# ### To check the train data we are using

# In[30]:


X_train


# In[24]:


from sklearn.linear_model import LogisticRegression


# ### Since the Logistic Regressioon imported is a class, we need to create an object of the class

# In[31]:


model = LogisticRegression()


# In[32]:


model.fit(X_train, y_train)


# In[33]:


## then we use our model to make predictions


# In[34]:


model.predict(X_test)


# ### To check the accuracy of the model, how it well predicted the answer

# In[35]:


model.score(X_test, y_test)


# ### To predict the probability of x_test

# In[36]:


model.predict_proba(X_test)


# In[ ]:




