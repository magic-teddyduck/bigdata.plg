#!/usr/bin/env python
# coding: utf-8

# In[47]:


#1(1)
from sklearn import datasets
iris = datasets.load_iris()
iris


# In[48]:


#1(2)
import pandas as pd
X = pd.DataFrame(iris["data"], columns = iris["feature_names"])
X.head(5)


# In[49]:


#1(3)
Y = pd.DataFrame(iris["target"], columns = ["target"])
Y.head(5)


# In[50]:


#1(4)
dfa4 = pd.concat([X,Y],axis=1)
dfa4.head(5)


# In[51]:


#1(5)
dfa5 = dfa4[(dfa4["target"] == 0)|(dfa4["target"] == 1)]
dfa5.tail(5)


# In[52]:


#1(6)
dfa5["target_class"] = dfa5["target"].apply(lambda x:1 if x ==0 else -1)
dfa6 = dfa5.drop(["target"],axis=1)
dfa6.tail(5)


# In[53]:


#1(7)
def sign(z):
    if z > 0:
        return 1
    else:
        return -1
import numpy as np
w = np.array([0.,0.,0.,0.,0.])
error = 1
iteration = 0
while error != 0:
    error = 0
    for i in range(len(dfa6)):
        x,y = np.concatenate((np.array([1.]),np.array(dfa6.iloc[i])[:4])),np.array(dfa6.iloc[i])[4]
        if sign(np.dot(w,x)) != y:
            iteration = iteration + 1    
            error = error + 1
            w=w + y*x
print("iteration:" + str(iteration) + " , " + "w:" + str(w)) 


# In[54]:


#1(8)
def sign(z):
    if z > 0:
        return 1
    else:
        return -1
import numpy as np
w=np.array([8.,8.,8.,8.,8.])
error=1
iteration=0
while error != 0:
    error=0
    for i in range(len(dfa6)):
        x=np.concatenate((np.array([1.]),np.array(dfa6.iloc[i])[:4]))
        y=np.array(dfa6.iloc[i])[4]
        if sign(np.dot(w,x)) != y:
            iteration += 1    
            error += 1
            w += y*x
print("iteration:" + str(iteration) + " , " + "w:" + str(w)) 


# In[55]:


#1(9)
newflower = np.array([1. ,3.0,2.1,4.6,3.3]) 
newflower = sign(np.dot(w,newflower))
if newflower == 1:
    print("是Setosa")
else:
    print("是Versicolor")


# In[56]:


#2(1)
dtf = pd.read_csv("titanic.csv")
dtf.head(20)


# In[57]:


#2(2)
from sklearn.impute import SimpleImputer
dtf1=dtf.fillna(value=dtf['Age'].median())
dtf1.head(20)


# In[58]:


#2(3)
dtf2=dtf1.replace({'1st':1,'2nd':2,'3rd':3})
dtf2.head(20)


# In[59]:


#2(4)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X = dtf2[['PClass','Age']]
Y = dtf2[['Survived']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)
X_train.head(20)


# In[60]:


#2(5)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,Y_train)
proba=clf.predict_proba(X_test)
for i,p in enumerate(proba):
    print(f"測試數據點{i}:不存活概率{p[0]},存活概率為{p[1]}")


# In[70]:


#2(6)
from sklearn.metrics import accuracy_score
Y_predict=clf.predict(X_test)
accuracy=accuracy_score(Y_test,Y_predict)
n_errors=len(Y_test)-accuracy_score(Y_test,Y_predict,normalize=False)
n_errors


# In[68]:


#2(7)
from sklearn.metrics import confusion_matrix
Y_predict=clf.predict(X_test)
cm=confusion_matrix(Y_test,Y_predict)
cm


# In[71]:


#2-8
from sklearn.metrics import accuracy_score
Y_predict=clf.predict(X_test)
accuracy=accuracy_score(Y_test,Y_predict)
print(f'{accuracy:.4f}')


# In[74]:


#2-9
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
clf=LogisticRegression()
clf.fit(X_train,Y_train)
Y_predict=clf.predict(X_test)
accuracy=accuracy_score(Y_test,Y_predict)
precision=precision_score(Y_test,Y_predict)
recall=recall_score(Y_test,Y_predict)
f1=f1_score(Y_test,Y_predict)
print(f"accuracy:{accuracy:.4f}")
print(f"precision:{precision:.4f}")
print(f"recall:{recall:.4f}")
print(f"f1:{f1:.4f}")


# In[75]:


#3(1)
dfb1 = pd.read_csv("pokemon.csv",encoding = "big5")
dfb1


# In[76]:


#3(2)
dfb2 = dfb1[dfb1["Attack"].notnull()]
dfb2 = dfb2[dfb2["Defense"].notnull()]
dfb2


# In[77]:


#3(3)
dfb3 = dfb2[(dfb2["Type"] == "Fighting")|(dfb2["Type"] == "Ghost")|(dfb2["Type"] == "Normal")]
dfb3


# In[78]:


#3(4)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = pd.DataFrame(dfb3.loc[:,"Attack":"Defense"])
sc.fit(X)
dfb4 = pd.DataFrame(sc.transform(X),columns = ["Attack","Defense"])
dfb4


# In[79]:


#3(5)
dfb5 = pd.DataFrame(dfb3["Type"].replace({"Fighting":0,"Ghost":1,"Normal":2}))
dfb5


# In[80]:


#3(6)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
check = OneVsOneClassifier(SVC(kernel = "rbf"))
check.fit(dfb4.values,dfb5.values)
error = 0
for i, v in enumerate(check.predict(dfb4.values)):
    if v != dfb5.values[i]:
        error += 1
print("error:" + str(error))


# In[81]:


#3(7)
from sklearn.metrics import confusion_matrix
predict = check.predict(dfb4.values)
confusion_matrix(dfb5,predict)


# In[82]:


#3(8)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(dfb5,predict)
print("Accuracy:"+str(accuracy.round(4)))


# In[83]:


#3(9)
from sklearn.metrics import f1_score
f1 = f1_score(dfb5,predict,average = "weighted")
print("F1-score:"+str(f1.round(4)))


# In[87]:


#3(10)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,test_idx = None,resolution = 0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))  
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
    Z = Z.reshape(xx1.shape)                                       
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)                
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x = X[y == cl,0],y = X[y == cl,1],alpha = 0.6,c = cmap(idx),edgecolor = "black",marker = markers[idx],label = cl)
plot_decision_regions(dfb4.values,dfb5["Type"].values,classifier = check)
plt.legend(loc = "upper left")
plt.tight_layout()
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.show()


# In[ ]:




