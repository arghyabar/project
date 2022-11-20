#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("iris_dataset.csv")                #LOADING THE DATASET


# In[3]:


df.head()                             #PRINTING FIRST FIVE ROWS            


# In[4]:


df.tail()                             #PRINTING LAST FIVE ROWS   


# In[5]:


df.columns                   #PRINT ONLY COLUMN NAMES


# In[6]:


pd.isnull(df)               #RETURNS TRUE FOR NULL VALUES OTHERWISE FALSE


# In[7]:


df.dropna()                  #REMOVE ALL ROWS WITH NULL VALUES


# In[8]:


df.info()                #PRINTS INFORMATION ABOUT THE DATAFRAME


# In[9]:


df.describe()                #OBTAINING STATISTICAL INFORMATION


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[11]:


sns.pairplot(df)                 #PAIRPLOTTING TO VIEW RELATIONSHIP


# In[12]:


sns.pairplot(df, hue="Species")              #TO GET THE DIFFERENCE IN VARIABLES TO MAP PLOT ASPECTS TO DIFFERENT COLORS


# In[13]:


sns.pairplot(df, diag_kind="kde")              #CHANGING THE DIAGONAL FROM HISTOGRAM TO KDE


# In[14]:


plt.figure(figsize=(5,5))                              #CHECK CORRELATION USING HEATMAP
sns.heatmap(df.corr(), annot=True, cmap= 'coolwarm')


# In[15]:


df.Species.value_counts()


# LinearRegression

# In[16]:


data=df.values
x=data[:,0:4]
y=data[:,:4]


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=2)


# In[20]:


l=LinearRegression()
l.fit(x_train,y_train)


# In[21]:


l.predict(x_test)


# In[22]:


plt.scatter(df['Petal length'],df['Petal Width'])
plt.plot(x_train,l.predict(x_train),color='red')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')


# In[23]:


l.coef_


# In[24]:


l.intercept_


# LogisticRegression

# In[25]:


f_mapping={'setosa':0,'versicolor':1,'virginica':2}           #CONVERTING CATEGORICAL VARIABLES INTO NUMBERS
df['Species']=df['Species'].map(f_mapping)


# In[26]:


x=df[['Petal length','Petal Width','Sepal Length','Sepal Width']].values
y=df[['Species']].values


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


model=LogisticRegression()


# In[29]:


model.fit(x,y)


# In[30]:


expected=y
predicted=model.predict(x)
predicted


# In[31]:


from sklearn.metrics import confusion_matrix
confusion_matrix(expected, predicted)


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(expected, predicted)

