#Stasya-Pupsik-Sladupsik

# отключим всякие предупреждения Anaconda
import warnings

warnings.filterwarnings("ignore") #игнорируем предупреждения, чтобы очистить консоль от лишнего (и самому не бояться :) )
import numpy as np
import pandas as pd
import pylab as plt 
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score #GridSearchCV - регулировка параметров дерева в поисках лучшего сочетания
from sklearn.neighbors import KNeighborsClassifier #kNN - k ближайших соседей
from sklearn.tree import DecisionTreeClassifier #Дерево решений, которое классивицирует объекты
from sklearn.metrics import accuracy_score #вычисление точности долей правильных ответов
from sklearn.pipeline import Pipeline #через него настроим шаги для обработки данных
from sklearn.preprocessing import StandardScaler #стандартизация всех параметров по формуле (x - mean) / std
from sklearn.ensemble import RandomForestClassifier #лес деревьев-классификаторов


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

'''  Обычные обучалки до того, как мы нашли наилучшие параметры
tree.fit(X_train, y_train) #обучаем дерево. X - параметры объектов, y - нужный результат
knn.fit(X_train, y_train) #обучаем соседей

tree_pred = tree.predict(X_holdout) #tree_pred - список классификации от X_holdout
print("Tree`s result:", accuracy_score(y_holdout, tree_pred)) #сравниваем список правильных классификаций и то, что получилось

knn_pred = knn.predict(X_holdout)
print("kNN`s result:", accuracy_score(y_holdout, knn_pred))
'''

#Тут ищем наилучшие параметры дерева решений и kNN
''' 
tree_params = {"max_depth": range(1, 11), "max_features": range(4, 19)} #указываем какие параметры дерева будем пробовать
                                                                        #глубину от 1 до 11, количество параметров - от 4 до 19
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True) #дерево, по каким параметрам смотрим, какая глубина кросс-валидации
                                                                           #работать на всех ядрах процессора и сразу всё в консоль
tree_grid.fit(X_train, y_train) #обучаем уже эту штуку, оно будет обучаться на всех параметрах, находя лучшее сочетание
print("Лучшие параметры у дерева:", tree_grid.best_params_, #обучили, теперь смотрим, какие лучшие параметры и лучший результат у дерева
      "\nЛучший результат:", tree_grid.best_score_) 
print('') #просто разделитель вывода в консоль
knn_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))])
#настройка «конвейера» - через какие функции будут проходить наши параметры
knn_params = {"knn__n_neighbors": range(1, 10)} #в параметрах указываем «пробежаться» по числу соседей от 1 до 10 в knn внутри Pipe
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True) #knn_pipe - то, где мы будем искать идеальные параметры, всё остальное по аналогии
knn_grid.fit(X_train, y_train) #обучение
print("Лучшие параметры у kNN:", knn_grid.best_params_,
      "\nЛучший результат:", knn_grid.best_score_)
'''

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17) #делаем лес деревьев. 100 деревьев, все процессора
forest_params = {"max_depth": range(1, 11), "max_features": range(4, 19)} #параметры настройки леса деревьев
forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True) #машинка, пробегающая по всем возможным параметрам леса в поисках наилучших
forest_grid.fit(X_train, y_train) #подача данных этой машинке для получение наилучших результатов

print("Лучшие параметры для леса:", forest_grid.best_params_,
      "Лучший показатель:", forest_grid.best_score_) #выведет средний результат по пяти тестам кросс-валидации