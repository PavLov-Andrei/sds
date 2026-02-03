import collections
from io import StringIO

import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from ipywidgets import Image
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)

def create_df(dic, feature_list):
    out = pd.DataFrame(dic) #создаём DataFrame из словаря
    out = pd.concat([out, pd.get_dummies(out[feature_list], drop_first=True)], axis=1) #делаем доп столбцы и удаляем один столбец dummy-данных для экономии места
    out.drop(feature_list, axis=1, inplace=True) #удалям оригиналы (по типу Sex, City и т.д.)
    return out

def intersect_features(train, test): #после обработки dummy-столбцов могли появиться разные данные (по типу Чикаго есть только в тестовых, а Майями есть только в обучающих)
    common_feat = list(set(train.keys()) & set(test.keys())) #находит пересечение множеств названий данных и преобразует в список
    return train[common_feat], test[common_feat]


features = ["Looks", "Alcoholic_beverage", "Eloquence", "Money_spent"] #список параметров
df_train = {} #создаем словарь, это будет будующая бдшка
df_train["Looks"] = ["handsome","handsome","handsome","repulsive","repulsive","repulsive","handsome",] #заполняем-заполняем-заполняем
df_train["Alcoholic_beverage"] = ["yes", "yes", "no", "no", "yes", "yes", "yes"]
df_train["Eloquence"] = ["high", "low", "average", "average", "low", "high", "average"]
df_train["Money_spent"] = ["lots", "little", "lots", "little", "lots", "lots", "lots"]
df_train["Will_go"] = LabelEncoder().fit_transform(["+", "-", "+", "-", "-", "+", "+"]) #тут команды изучают уникальные признаки и делят их на 0 и 1
df_train = create_df(df_train, features) #превращаем словарь в DataFrame

df_test = {} #то же самое, но для тестовых данных
df_test["Looks"] = ["handsome", "handsome", "repulsive"]
df_test["Alcoholic_beverage"] = ["no", "yes", "yes"]
df_test["Eloquence"] = ["average", "high", "average"]
df_test["Money_spent"] = ["lots", "little", "lots"]
df_test = create_df(df_test, features)

y = df_train["Will_go"] #т.к. этого столбца нет в тренировочных данных, мы его заранее запоминаем
df_train, df_test = intersect_features(train=df_train, test=df_test) #оставляем только одинаковые столбцы
#print(df_test.keys(), '\n', df_train.keys()) #т.к. мы удаляли некоторые столбцы в функции создания df, нужно проверить, всё ли нужное на месте
#строчка сверху показало, что всё норм