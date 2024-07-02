#!/usr/bin/env python
# coding: utf-8

# In[72]:


#1-1
import pandas as pd
df1=pd.read_csv("wine.csv")
df1.head(5)


# In[73]:


#1-2
X=df1.drop(["Target"],axis=1)
Y=df1["Target"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=42)
X_train.head(10)


# In[74]:


#1-3
from sklearn.tree import DecisionTreeClassifier
df2= DecisionTreeClassifier(criterion = 'entropy', max_depth=5,random_state=42)
df2.fit(X_train,Y_train)
error = 0
for i, v in enumerate(df2.predict(X_test)):
    if v!= Y_test.values[i]:
        error+=1
print(error)


# In[75]:


#1-4
from sklearn.metrics import confusion_matrix
y_pred = df2.predict(X_test)
confusion_matrix_result = confusion_matrix(Y_test, y_pred)
print(confusion_matrix_result)


# In[76]:


#1-5
accuracy=df2.score(X_test,Y_test).round(4)
print(accuracy)


# In[77]:


#1-6
from sklearn.metrics import classification_report
y_pred2 = df2.predict(X_test)
report = classification_report(Y_test, y_pred2, digits=4)
print(report)


# In[96]:


#1-7
from sklearn.tree import export_graphviz
from graphviz import Source
target_names = ['class 1','class 2','class 3']
export_graphviz(df2, out_file='tree.dot', feature_names=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD of diluted wines','Proline'], class_names=target_names)
Source.from_file('tree.dot')


# In[55]:


#1-8
new=pd.DataFrame([[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]
,columns = ["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD of diluted wines","Proline"])
if (df2.predict(new) == 1):
    print("第1類")
elif (df2.predict(new) == 2):
    print("第2類")
elif (df2.predict(new) == 3):
    print("第3類")


# In[57]:


#1-9
from sklearn.ensemble import RandomForestClassifier
df3 = RandomForestClassifier(criterion='entropy',n_estimators=100, random_state=42, n_jobs=2)
df3.fit(X_train, Y_train)
error = 0
for i, v in enumerate(df3.predict(X_test)):
    if v!= Y_test.values[i]:
        error+=1
print(error)


# In[58]:


#1-10
accuracy=df3.score(X_test,Y_test).round(4)
print(accuracy)


# In[59]:


#1-11
print('隨機森林')


# In[80]:


#2-1
import pandas as pd
df4=pd.read_csv("Mall_Customers.csv")
df4.head(10)


# In[83]:


#2-2
import matplotlib.pyplot as plt
x=df4['Annual Income (k$)'].values
y=df4['Spending Score (1-100)'].values
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.scatter(x,y)
plt.show()


# In[85]:


#2-3
df5=df4[['Annual Income (k$)','Spending Score (1-100)']]
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=5,random_state=0)
model=cluster.fit(df5)
model.cluster_centers_


# In[86]:


#2-4
df4['cluster']=model.labels_
df4.head(10)


# In[92]:


#2-5
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state=0, n_jobs=-1)
    kmeans.fit(df5)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[94]:


#2-6
print('合理')


# In[95]:


#2-7
a = df5.to_numpy()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, random_state = 0)
kmeans.fit(a)
y_kmeans = kmeans.fit_predict(a)

plt.scatter(a[y_kmeans == 0, 0], a[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(a[y_kmeans == 1, 0], a[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(a[y_kmeans == 2, 0], a[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(a[y_kmeans == 3, 0], a[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(a[y_kmeans == 4, 0], a[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:, 0],  kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




