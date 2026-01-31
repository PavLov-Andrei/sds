#Stasya-Pupsik-Sladupsik

# отключим всякие предупреждения Anaconda
import warnings

warnings.filterwarnings("ignore") #игнорируем предупреждения, чтобы очистить консоль от лишнего (и самому не бояться :) )
import numpy as np
import pandas as pd
import pylab as plt 
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier #kNN - k ближайших соседей
from sklearn.tree import DecisionTreeClassifier #Дерево решений, которое классивицирует объекты
from sklearn.metrics import accuracy_score #вычисление точности долей правильных ответов

df = pd.read_csv("../mlcourse.ai/data/telecom_churn.csv") #Считываем базу данных в DataFrame
df["International plan"] = pd.factorize(df["International plan"])[0] #переводим все значения в 0 и 1
df["Voice mail plan"] = pd.factorize(df["Voice mail plan"])[0] #factorize делает ([1, 1, 0], ["Yes", "Yes", "No"])
df["Churn"] = df["Churn"].astype("int")  #переводим всё из булевского в int 
states = df["State"]
y = df["Churn"]
df.drop(["State", "Churn"], axis=1, inplace=True) #удаляем из df Штаты и Ушедних/Оставшихся клиентов, меняя оригинальный df
#print(df.head())

X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)
#X_train, y_train - тренировочные, X_holdout, y_holdout - тестовые

tree = DecisionTreeClassifier(max_depth=5, random_state=17) #пока не знаем хороших параметров, берём 5
knn = KNeighborsClassifier(n_neighbors=10) #пока не знаем хороших параметров, берём наугад

tree.fit(X_train, y_train) #обучаем дерево. X - параметры объектов, y - нужный результат
knn.fit(X_train, y_train) #обучаем соседей

tree_pred = tree.predict(X_holdout) #tree_pred - список классификации от X_holdout
print("Tree`s result:", accuracy_score(y_holdout, tree_pred)) #сравниваем список правильных классификаций и то, что получилось

knn_pred = knn.predict(X_holdout)
print("kNN`s result:", accuracy_score(y_holdout, knn_pred))