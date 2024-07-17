import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model
pokemon=pd.read_csv("F:\BaiduNetdiskDownload\Pokemon.csv")
pokemon.head()
pokemon.info()#查看表头和列
label_encoder=LabelEncoder()
pokemon['Legendary']=label_encoder.fit_transform(pokemon['Legendary'])
#print(pokemon)
pokemon['Generation'].value_counts().sort_index().plot.bar(rot=0,xlabel='Generation',ylabel='Num')#柱状图表示每代精灵的数量1
# t = pokemon["Generation"].value_counts().sort_index()
# plt.bar(t.index, t.values)
plt.show()
generation_legendary_num=pokemon.groupby(by='Generation')['Legendary'].sum()
generation_legendary_num.plot(kind='bar',rot=0,xlabel='Generation',ylabel='Legendary')
plt.show()
generation_legendary_type=pokemon.groupby(by='Type 1')['Legendary'].sum()
generation_legendary_type.plot.pie(autopct='%1.1f',figsize=(8,8))
plt.show()
num1=pokemon['Legendary'].sum()#传说精灵总数
num2=pokemon[(pokemon['Legendary']==True)&(pokemon['Type 2'].notnull())].shape[0]#有第二属性的传说精灵数
num3=pokemon['Type 2'].notnull().sum()#有第二属性的精灵数
num4=num1-num2  #只具备第一属性的传说精灵数量
num5=800-num3  #只具备第一属性的精灵数量
rate1=num4/num5  #只具备第一属性传说精灵占比
rate2=num2/num3  #具备第二属性传说精灵占比
if(-0.01<rate1-rate2<0.01):
    print("传说精灵与是否具备第二属性无关")
else:
    print("传说精灵与是否具备第二属性有关")
X=pokemon.iloc[:,2:-1]
Y=pokemon['Legendary']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)