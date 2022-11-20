import pandas as pd
import numpy as np

df=pd.read_csv("iris_dataset.csv")                #LOADING THE DATASET

df.head()                             #PRINTING FIRST FIVE ROWS            

df.tail()                             #PRINTING LAST FIVE ROWS   

df.columns                   #PRINT ONLY COLUMN NAMES

pd.isnull(df)               #RETURNS TRUE FOR NULL VALUES OTHERWISE FALSE

df.dropna()                  #REMOVE ALL ROWS WITH NULL VALUES

df.info()                #PRINTS INFORMATION ABOUT THE DATAFRAME

df.describe()                #OBTAINING STATISTICAL INFORMATION

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.pairplot(df)                 #PAIRPLOTTING TO VIEW RELATIONSHIP

sns.pairplot(df, hue="Species")              #TO GET THE DIFFERENCE IN VARIABLES TO MAP PLOT ASPECTS TO DIFFERENT COLORS

sns.pairplot(df, diag_kind="kde")              #CHANGING THE DIAGONAL FROM HISTOGRAM TO KDE

plt.figure(figsize=(5,5))                              #CHECK CORRELATION USING HEATMAP
sns.heatmap(df.corr(), annot=True, cmap= 'coolwarm')

df.Species.value_counts()

# LinearRegression

data=df.values
x=data[:,0:4]
y=data[:,:4]

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=2)

l=LinearRegression()
l.fit(x_train,y_train)

l.predict(x_test)
plt.scatter(df['Petal length'],df['Petal Width'])
plt.plot(x_train,l.predict(x_train),color='red')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')

l.coef_

l.intercept_

# LogisticRegression

f_mapping={'setosa':0,'versicolor':1,'virginica':2}           #CONVERTING CATEGORICAL VARIABLES INTO NUMBERS
df['Species']=df['Species'].map(f_mapping)

x=df[['Petal length','Petal Width','Sepal Length','Sepal Width']].values
y=df[['Species']].values

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(x,y)

expected=y
predicted=model.predict(x)
predicted

from sklearn.metrics import confusion_matrix
confusion_matrix(expected, predicted)

from sklearn.metrics import accuracy_score
accuracy_score(expected, predicted)

