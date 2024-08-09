import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn import metrics

ds = pd.read_csv("iris.csv").to_numpy()
for n, val in enumerate(ds):
    if val[4] == "Iris-setosa":
        ds[n][4] = 0
    elif val[4] == "Iris-versicolor":
        ds[n][4] = 1
    else:
        ds[n][4] = 2

[inp, outp] = np.split(ds, [4], axis=1)
outp = np.array([elem[0] for elem in outp])
x_train, x_test, y_train, y_test = train_test_split(inp, outp, test_size=0.15)
km = KMeans(n_clusters=3)
km = km.fit(x_train, y_train)
pred = km.predict(x_test)

print(y_test)
print(pred)
print()
print('Erro Absoluto:', metrics.mean_absolute_error(y_test, pred))
print('Erro Quadrado:', metrics.mean_squared_error(y_test, pred))
print('Raiz do Erro Quadrado:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print()
print('Score Validação Cruzada:', cross_val_score(km, x_test, y_test, cv=5))
print('Score Modelo:', km.score(x_test, y_test))
