#!/usr/bin/env python
# coding: utf-8

# In[3]:


#3-1
import pandas as pd
import numpy as np

df=pd.read_csv("Automobile_data.csv")
print(df)


# In[4]:


#3-2
print(pd.merge(df.head(4),df.tail(4),how="outer"))


# In[5]:


#3-3
print(df[df["price"]==df.loc[:,"price"].max()])


# In[6]:


#3-4
print(df[df['company']=="toyota"])


# In[7]:


#3-5
print(pd.DataFrame(df.groupby("company").size(),columns=["車輛總數"]))


# In[8]:


#3-6
print(pd.DataFrame(df.groupby("company")["average-mileage"].mean()))


# In[9]:


#3-7
print(df.sort_values("price",ascending=False))


# In[10]:


#3-8
mapping={"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12}
df["num-of-cylinders"]=df["num-of-cylinders"].map(mapping)
print(df)


# In[11]:


#3-9
df["ratio-of-price_cylinders"]=df["price"]/df["num-of-cylinders"]
print(df)


# In[12]:


#3-10
df_audi=df[(df['company']=="audi")&(df['body-style']=="sedan")]
df_bmw=df[(df['company']=="bmw")&(df['body-style']=="sedan")]
df_merge=pd.concat([df_audi,df_bmw])
print(df_merge)


# In[ ]:




