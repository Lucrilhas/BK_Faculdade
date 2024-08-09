import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

ds_att = pd.read_csv("IA_data/concreto_att.csv")

# Normalizar Dataset com Z-Score
ds = ds_att.copy()
for COL in ds.columns:
    ds[COL] = (ds[COL] - ds[COL].mean()) / ds[COL].std()

x = ds.drop(columns='concrete_compressive_strength')
y = ds['concrete_compressive_strength']

reg = LinearRegression()
reg.fit(x, y)

pickle.dump(reg, open("IA_data/linearRegressionConcreteAtt.sav", 'wb'))