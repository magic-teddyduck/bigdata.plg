#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2-1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImpute
df=pd.read_csv("titanic.csv")
print(df)


# In[2]:


#2-2
survived=df.loc[:,'Survived'].mean()
print('存活比率=',survived*100,'%')


# In[3]:


#2-3
print(pd.DataFrame(df.isnull().sum(),columns=['空缺值']))


# In[4]:


#2-4
df4=df.drop('Cabin',axis=1)
print(df4)


# In[5]:


#2-5
print(df4[df4['Age'].isnull()])


# In[6]:


#2-6
import pandas as pd
import numpy as np

X=df4.iloc[:].values
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp=imp.fit(X[:,[5]])
X[:,[5]]=imp.transform(X[:,[5]])
df6=pd.DataFrame(X)
df6.columns=['PassengerId', 'Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']
print(df6)


# In[7]:


#2-7
X=df6.iloc[:].values
imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp=imp.fit(X[:,[10]])
X[:,[10]]=imp.transform(X[:,[10]])

df7=pd.DataFrame(X)
df7.columns=['PassengerId', 'Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']
print(df7)


# In[8]:


#2-8
print(pd.DataFrame(df7.groupby('Sex')['Age'].mean()))


# In[9]:


#2-9
df7.loc[(df7['Sex']=='male')&(df7['Age']<12),'Fare']=df7.loc[(df7['Sex']=='male')&(df7['Age']<12),'Fare']*0.8
print(df7['Fare'])


# In[10]:


#2-10
df7['Fare']=df7['Fare'].astype(float)
df7.loc[df7['Age']>40,'Fare']=np.round(df7.loc[(df7['Age']>40),'Fare']*0.9,1)
print(df7['Fare'])


# In[ ]:




