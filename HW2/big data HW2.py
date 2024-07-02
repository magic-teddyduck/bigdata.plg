#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
#1-(1)
df=pd.read_excel('Taiwan_ele_data.xlsx')
print(df)


# In[9]:


#1-(2)
plt.style.use('ggplot')
font=FontProperties(fname=r'C:\Users\vince\AppData\Local\Microsoft\Windows\Fonts\SimHei.ttf')
plt.plot(df['年'],df['總用電量'],color='red',linestyle='--',linewidth='1')
plt.title('我國電力年消費量變化趨勢',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.xlim(2004, 2022) 
plt.show()


# In[10]:


#1-(3)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [9, 7]
plt.plot(df['年'],df['能源部門自用'],label='能源部門自用')
plt.plot(df['年'],df['工業部門'],label='工業部門')
plt.plot(df['年'],df['運輸部門'],label='運輸部門')
plt.plot(df['年'],df['農業部門'],label='農業部門')
plt.plot(df['年'],df['服務業部門'],label='服務業部門')
plt.plot(df['年'],df['住宅部門'],label='住宅部門')
plt.title('我國各部門電力年消費量變化趨勢',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('各部門用電量',fontproperties=font)
plt.xlim(2004, 2022) 
plt.legend(loc=2,prop=font)
plt.show()


# In[11]:


#1-(4)
plt.rcParams['figure.figsize'] = [6.4, 4.8]
plt.scatter(df['年'],df['住宅部門'],label='住宅部門')
plt.title('住宅部門',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.grid(linestyle='--')
plt.xlim(2004, 2022) 
plt.legend(loc=2,prop=font)
plt.show()


# In[12]:


#1-(5)
plt.bar(df['年']-0.2,df['服務業部門'],label='服務業部門',width=0.4)
plt.bar(df['年']+0.2,df['住宅部門'],label='住宅部門',width=0.4)
plt.title('服務業部門與住宅部門年用電量比較',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.legend(loc=2,prop=font)
plt.show()


# In[13]:


#1-(6)
df6=pd.DataFrame(df.loc[:,:].sum())
df6=df6.drop('年',axis=0)
df6=df6.drop('總用電量',axis=0)
plt.pie(df6.iloc[:,0],autopct='%1.1f%%',radius=1.5,labels=df6.index,textprops={'fontproperties':font})
plt.show()


# In[14]:


#1-(7)
df7=pd.DataFrame(df.loc[:,:].sum())
df7=df7.drop('年',axis=0)
df7=df7.drop('總用電量',axis=0)
separated=(0,0,0,0,0,0.2)
plt.pie(df7.iloc[:,0],autopct='%1.1f%%',radius=1.5,labels=df7.index,textprops={'fontproperties':font},explode=separated)
plt.show()


# In[15]:


#1-(8)
plt.subplot(311)
plt.plot(df['年'],df['工業部門'],color='green')
plt.title('工業部門年用電資料',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.xlim(2004, 2022)
plt.subplot(312)
plt.plot(df['年'],df['服務業部門'],color='red')
plt.title('服務業部門年用電資料',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.xlim(2004, 2022)
plt.subplot(313)
plt.plot(df['年'],df['住宅部門'],color='blue')
plt.title('住宅部門年用電資料',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.xlim(2004, 2022)
plt.show()


# In[18]:


#1-(9)
df=df.T
df.columns=df.iloc[0]
df=df.drop(df.iloc[0].index.name)
df.columns.name =None
df=df.T
df.iloc[:,1:7].plot(kind='area')
plt.title('各部門年用電量之堆疊圖',fontproperties=font)
plt.xlabel('年',fontproperties=font)
plt.ylabel('每年用電量',fontproperties=font)
plt.legend(loc=2,prop=font)
plt.xlim(2004,2022)
plt.show()


# In[19]:


#2-(1)
data = {'Book ID': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'],
        'Income': [1000, 3000, 30000, 24000, 300, 400, 800, 2300, 12000, 1800]}

df = pd.DataFrame(data)
print(df)


# In[20]:


#2-(2)
df = df.sort_values(by='Income', ascending=False)
df = df.reset_index(drop=True)
print(df)


# In[21]:


#2-(3)
total_income = df['Income'].sum()
df['Income_percent'] = df['Income'] / total_income
print(df)


# In[22]:


#2-(4)
df['Income_cum_percent'] = df['Income_percent'].cumsum()
print(df)


# In[23]:


#2-(5)
font=FontProperties(fname=r'C:\Users\vince\AppData\Local\Microsoft\Windows\Fonts\SimHei.ttf')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
df.plot(kind='bar', x='Book ID', y='Income', ax=ax1, color='blue')
ax1.set_ylabel('收入',fontproperties=font)
df.plot(x='Book ID', y='Income_cum_percent', ax=ax2, color='red', marker='o')
ax2.set_ylim([0, 1])
ax2.set_ylabel('收入累積比例',fontproperties=font)
plt.title('收入主次因素分析',fontproperties=font)
ax1.set_xlabel('圖書ID',fontproperties=font)
ax1.set_xticklabels(df['Book ID'], rotation=0)
plt.show()


# In[24]:


#3-(1)
df=pd.read_excel('health.xlsx')
print(df)


# In[25]:


#3-(2)
df2=df
df2['每單位卡路里之行走步數']=df['每日行走步數']/df['攝取卡路里']
print(df2)


# In[28]:


#3-(3)
df3=df2
condlist=[(df3['每單位卡路里之行走步數']<=2),(df3['每單位卡路里之行走步數']>2)&(df3['每單位卡路里之行走步數']<=4.5),(df3['每單位卡路里之行走步數']>4.5)]
choicelist=['低','中','高']
df3['健康指數']=np.select(condlist,choicelist,default='Not Specified')
print(df3)


# In[29]:


#3-(4)
df4=pd.get_dummies(df2['健康指數'],prefix='健康指數')
df3['健康指數_高']=df4['健康指數_高']
df3['健康指數_中']=df4['健康指數_中']
df3['健康指數_低']=df4['健康指數_低']
print(df3)


# In[ ]:




