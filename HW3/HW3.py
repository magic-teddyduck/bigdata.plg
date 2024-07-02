#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1(1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv('Taipei_house.csv')
df1


# In[4]:


#1(2)
pd.DataFrame(df1.isnull().sum())


# In[6]:


#1(3)
One_Hot_Encoding=pd.get_dummies(df1['行政區'],prefix='行政區')
One_Hot_Encoding
a = pd.concat([df1,One_Hot_Encoding],axis=1)
df3 = a.drop("行政區", axis = 1)
df3


# In[7]:


#1(4)
df4 = df3.replace({'無':0, '坡道平面':1, '坡道機械':1, '升降機械':1,
                    '升降平面':1,'其他':1, '塔式車位':1, '一樓平面':1})
df4


# In[10]:


#1(5)
df5 = df1.corr(method = "pearson")
df5 = pd.DataFrame(df5["總價"]).round(2).drop(index = "總價")
df5


# In[11]:


#1(6)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X = df4[['建物總面積']]
y = df4[['總價']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42) 
X_train.head(5)


# In[13]:


#1(7)
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X_train,Y_train)
rs = lr.score(X_train,Y_train)
print("截距項 =",lr.intercept_)
print("X係數 =",lr.coef_)
print("R平方 =",rs)


# In[16]:


#1(8)
X = df4[['土地面積','建物總面積','屋齡','樓層','總樓層','用途','房數','廳數','衛數','電梯','車位類別','行政區_信義區',
         '行政區_大安區','行政區_文山區','行政區_松山區']]
y = df4[['總價']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42) 


regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
r_squared = regr.score(X_train, y_train)

print('截距項', regr.intercept_)
print('X係數', regr.coef_)
print('R平方:',r_squared)


# In[18]:


#1(9)
print("均方根誤差(RMSE): %.4f" % np.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))


# In[19]:


#1(10)
print("絕對誤差(MAE): %.4f" % np.mean(abs(regr.predict(X_test) - y_test)))


# In[24]:


#1(11)
predict = regr.predict([[36,99,32,4,4,0,3,2,1,0,0,0,0,0,1]])
print('預測房屋總價: %.4f'% predict)


# In[25]:


#1(12)
def Ra2(r_squared,n,p):
    return 1-(1-r_squared)*((n-1)/(n-p))
n = len(X_test)
p = X_test.shape[1]

Ra2 = Ra2(r_squared, n, p)
print('調整後判定係數Ra2:', Ra2)


# In[26]:


#1(13)
from sklearn.preprocessing import PolynomialFeatures
plt.style.use('ggplot')

quadratic = PolynomialFeatures(degree=2) 
X_poly_train = quadratic.fit_transform(X_train)
X_poly_test = quadratic.fit_transform(X_test)   


regr = linear_model.LinearRegression()

regr.fit(X_poly_train, y_train)
r_squared = regr.score(X_poly_train, y_train)

print('截距項', regr.intercept_)
print('X係數', regr.coef_)
print('R平方:',r_squared)


# In[ ]:




