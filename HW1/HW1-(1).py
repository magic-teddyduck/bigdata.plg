#!/usr/bin/env python
# coding: utf-8

# In[6]:


#1-1
import pandas as pd
import numpy as np
df=pd.DataFrame([[65,72,78,93,56],
                 [90,92,91,83,70],
                 [81,53,76,94,94],
                 [79,85,47,88,80],
                 [70,61,32,70,88],
                 [88,82,99,68,79],
                 [91,76,87,51,67],
                 [55,64,62,78,52],
                 [40,46,55,60,71]],
                 index=["李國發","吳俊諺","蔡俊毅","姚鈺迪","袁劭彥","蔡登意","吳景翔","邱孝信","陳明輝"],
                 columns=["國文成績","英文成績","數學成績","自然成績","社會成績"])
print(df)


# In[7]:


#1-2
df2=df[df['國文成績']>80]
print(df2)


# In[15]:


#1-3
print(df.loc[['蔡俊毅','袁劭彥','吳景翔'],['英文成績','自然成績']])


# In[16]:


#1-4
print(df.iloc[[1,8],[0,4]])


# In[17]:


#1-5
print(df.iloc[3:,1:4])


# In[18]:


#1-6
print(df.sort_values(('數學成績'),ascending=True))


# In[21]:


#1-7
from scipy import stats
df=pd.DataFrame(df,dtype=float)
g=stats.gmean(df.iloc[:,:])
dfg=pd.DataFrame([[g[0]],[g[1]],[g[2]],[g[3]],[g[4]]],index=['國文成績','英文成績','數學成績','自然成績','社會成績'],
                 columns=['幾何平均數'])
print(dfg)


# In[22]:


#1-8
df['體育成績']=[65,75,71,69,70,98,81,59,70]
print(df)


# In[25]:


#1-9
df['體育成績']=[65,75,71,69,70,98,81,59,70]
file_name='HW1.xlsx'
df.to_excel(file_name)


# In[ ]:




