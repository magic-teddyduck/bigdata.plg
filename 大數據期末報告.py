#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 讀取CSV檔案
data = pd.read_csv("game_stats.csv")
data.head(6)


# In[163]:


import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#防守數據方面
# 載入資料集
data = pd.read_csv("game_stats.csv")
data.fillna(0, inplace=True)

# 分割特徵和標籤
X = data[['對手兩分%','對手三分%']]
y = data['勝率']

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立回歸模型
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
r_squared = model.score(X_train, y_train)
print('R平方值:',r_squared)

# 模型評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("MSE:",mse)
print("Root Mean Squared Error:", rmse)

# 計算VIF
X_train_with_constant = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Features"] = X_train_with_constant.columns
vif["VIF Factor"] = [sm.OLS(X_train_with_constant[col], X_train_with_constant.drop(col, axis=1))
                     .fit().rsquared for col in X_train_with_constant.columns]
vif = vif.sort_values(by="VIF Factor", ascending=False)

# 找出容忍值大於0.1的項目
tolerance = 1 / vif["VIF Factor"]
significant_features = vif[vif["VIF Factor"] < 0.1]["Features"]
print("Variables with tolerance > 0.1:")
print(significant_features)

# 繪製預測值和實際值的差異圖
plt.scatter(y_test, y_pred)
plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # 理想情況下的直線
plt.xlabel('Ture win rate')
plt.ylabel('Predict win rate')
plt.title('difference between ture & predict')
plt.show()


# In[164]:


#進攻數據方面
# 載入資料集
data = pd.read_csv("game_stats.csv")
data.fillna(0, inplace=True)

# 分割特徵和標籤
X = data[['兩分%','三分%']]
y = data['勝率']

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立回歸模型
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
r_squared = model.score(X_train, y_train)
print('R平方值:',r_squared)

# 模型評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("MSE:",mse)
print("Root Mean Squared Error:", rmse)

# 計算VIF
X_train_with_constant = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Features"] = X_train_with_constant.columns
vif["VIF Factor"] = [sm.OLS(X_train_with_constant[col], X_train_with_constant.drop(col, axis=1))
                     .fit().rsquared for col in X_train_with_constant.columns]
vif = vif.sort_values(by="VIF Factor", ascending=False)

# 找出容忍值大於0.1的項目
tolerance = 1 / vif["VIF Factor"]
significant_features = vif[vif["VIF Factor"] < 0.1]["Features"]
print("Variables with tolerance > 0.1:")
print(significant_features)

# 繪製預測值和實際值的差異圖
plt.scatter(y_test, y_pred)
plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # 理想情況下的直線
plt.xlabel('Ture win rate')
plt.ylabel('Predict win rate')
plt.title('difference between ture & predict')
plt.show()


# In[115]:


import pandas as pd

# 讀取CSV檔案
df = pd.read_csv('Win_stats.csv')

# 將'主/客場隊伍'中的資料轉換為數字
df['主場隊伍'] = df['主場隊伍'].map({'新北國王': 1, '台北富邦勇士': 0})
df['客場隊伍'] = df['客場隊伍'].map({'新北國王': 1, '台北富邦勇士': 0})

#將贏球隊伍中的資料轉為數字
df['贏球隊伍'] = df['贏球隊伍'].map({'新北國王': 1, '台北富邦勇士': 0})

# 創建新的欄位，初始值為0
df['主場或客場'] = 0

# 將主場贏球的場次的欄位設置為0
df.loc[(df['主場隊伍'] == 1) & (df['贏球隊伍'] == 1), '主場或客場'] = 0
df.loc[(df['主場隊伍'] == 0) & (df['贏球隊伍'] == 0), '主場或客場'] = 0

# 將客場贏球的場次的欄位設置為1
df.loc[(df['主場隊伍'] == 0) & (df['贏球隊伍'] == 1), '主場或客場'] = 1
df.loc[(df['主場隊伍'] == 1) & (df['贏球隊伍'] == 0), '主場或客場'] = 1


# 顯示處理後的資料
df.head(14)


# In[168]:


import pandas as pd
import numpy as np
from math import log2
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def calculate_entropy(labels):
    n = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    entropy = 0
    for count in counts:
        p = count / n
        entropy -= p * log2(p)
    return entropy

def calculate_information_gain_ratio(data, attribute, target):
    entropy_s = calculate_entropy(data[target])
    attribute_values, attribute_counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = 0
    intrinsic_value = 0
    for value, count in zip(attribute_values, attribute_counts):
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset[target])
        weighted_entropy += (count / len(data)) * subset_entropy
        intrinsic_value -= (count / len(data)) * log2(count / len(data))
    information_gain = entropy_s - weighted_entropy
    information_gain_ratio = information_gain / intrinsic_value
    return information_gain_ratio

def calculate_gini_index(data, attribute, target):
    gini_index = 0
    attribute_values, attribute_counts = np.unique(data[attribute], return_counts=True)
    for value, count in zip(attribute_values, attribute_counts):
        subset = data[data[attribute] == value]
        p = len(subset) / len(data)
        target_values, target_counts = np.unique(subset[target], return_counts=True)
        gini = 1
        for target_value, target_count in zip(target_values, target_counts):
            gini -= (target_count / len(subset)) ** 2
        gini_index += p * gini
    return gini_index

# 1. 計算資料增益比
attribute_list = ['主隊得分', '主隊兩分%', '主隊三分%','客隊得分', '客隊兩分%', '客隊三分%']
target_attribute = '主場或客場'
information_gains = []
for attribute in attribute_list:
    information_gain_ratio = calculate_information_gain_ratio(df, attribute, target_attribute)
    information_gains.append((attribute, information_gain_ratio))
max_information_gain_ratio = max(information_gains, key=lambda x: x[1])[0]
print("使用資料增益比進行預測的最佳屬性1：", max_information_gain_ratio)
attribute_list = ['主隊得分', '主隊兩分%', '主隊三分%','客隊得分',  '客隊三分%']
target_attribute = '主場或客場'
information_gains = []
for attribute in attribute_list:
    information_gain_ratio = calculate_information_gain_ratio(df, attribute, target_attribute)
    information_gains.append((attribute, information_gain_ratio))
max_information_gain_ratio = max(information_gains, key=lambda x: x[1])[0]
print("使用資料增益比進行預測的最佳屬性2：", max_information_gain_ratio)

# 2. 計算Gini
gini_indices = []
for attribute in attribute_list:
    gini_index = calculate_gini_index(df, attribute, target_attribute)
    gini_indices.append((attribute, gini_index))
max_gini_index = max(gini_indices, key=lambda x: x[1])[0]
print("使用GINI進行預測的最佳屬性1：", max_gini_index)
attribute_list = [ '主隊兩分%', '主隊三分%','客隊得分', '客隊兩分%', '客隊三分%']
gini_indices = []
for attribute in attribute_list:
    gini_index = calculate_gini_index(df, attribute, target_attribute)
    gini_indices.append((attribute, gini_index))
max_gini_index = max(gini_indices, key=lambda x: x[1])[0]
print("使用GINI進行預測的最佳屬性2：", max_gini_index)


# In[182]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
#設定目標
df1 = df[['主隊三分%','客隊兩分%','主場或客場']]
df1 = df[df['主場或客場'].isin([0,1])]

# 創建決策樹模型
X_train, X_test, y_train, y_test = train_test_split(df1[['主隊三分%','客隊兩分%']], df1[['主場或客場']], test_size=0.3,random_state=0)
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=5,random_state=0)
tree.fit(X_train,y_train)
tree.predict(X_test)
y_test['主場或客場'].values

#印出錯誤個數
error = 0
print('錯誤點位:')
for i, v in enumerate(tree.predict(X_test)):
    if v!= y_test['主場或客場'].values[i]:
        print(i,v)
        error+=1
print('錯誤個數:',error)

# 計算預測準確率
accuracy=tree.score(X_test,y_test['主場或客場'])
print('預測正確率:',accuracy)

# 取得訓練資料的特徵欄位
X = X_train[['主隊三分%', '客隊兩分%']].values
# 取得訓練資料的目標欄位
y = y_train['主場或客場'].values

# 訓練分類器
tree.fit(X, y)

# 定義繪圖範圍
plot_x_min, plot_x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
plot_y_min, plot_y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(plot_x_min, plot_x_max, 0.01),
                     np.arange(plot_y_min, plot_y_max, 0.01))

# 用分類器預測網格中每個點的類別
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製決策區域圖
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
plt.xlabel('home team three point%')
plt.ylabel('guest team two point%')
plt.title('Decision Boundary')
plt.show()

# 顯示決策樹圖形
export_graphviz(tree, out_file='tree.dot', feature_names=['home team three point%', 'guest team two point%'], class_names=['home', 'guest'], filled=True)
src = Source.from_file('tree.dot')
src.view() 


# In[183]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#設定目標
df2 = df[['主隊得分','客隊三分%','主場或客場']]
df2 = df[df['主場或客場'].isin([0,1])]

# 創建決策樹模型
X_train, X_test, y_train, y_test = train_test_split(df2[['主隊得分','客隊三分%']], df2[['主場或客場']], test_size=0.3,random_state=0)
tree = DecisionTreeClassifier(criterion = 'gini', max_depth=5,random_state=0)
tree.fit(X_train,y_train)
tree.predict(X_test)
y_test['主場或客場'].values

#印出錯誤個數
error = 0
print('錯誤點位:')
for i, v in enumerate(tree.predict(X_test)):
    if v!= y_test['主場或客場'].values[i]:
        print(i,v)
        error+=1
print('錯誤個數:',error)

# 計算預測準確率
accuracy=tree.score(X_test,y_test['主場或客場'])
print('預測正確率:',accuracy)

# 取得訓練資料的特徵欄位
X = X_train[['主隊得分', '客隊三分%']].values
# 取得訓練資料的目標欄位
y = y_train['主場或客場'].values

# 訓練分類器
tree.fit(X, y)

# 定義繪圖範圍
plot_x_min, plot_x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
plot_y_min, plot_y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(plot_x_min, plot_x_max, 0.01),
                     np.arange(plot_y_min, plot_y_max, 0.01))

# 用分類器預測網格中每個點的類別
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製決策區域圖
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
plt.xlabel('home team score')
plt.ylabel('guest three point%')
plt.title('Decision Boundary')
plt.show()

# 顯示決策樹圖形
export_graphviz(tree, out_file='tree.dot', feature_names=['home team score', 'guest three point%'], class_names=['home', 'guest'], filled=True)
src = Source.from_file('tree.dot')
src.view() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




