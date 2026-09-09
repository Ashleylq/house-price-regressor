from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/wblakecannon/ames/refs/heads/master/data/housing.csv")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

num = X.select_dtypes(include=["number"]).columns.tolist()
cat = X.select_dtypes(exclude=["number"]).columns.tolist()

num.remove("MS SubClass")
cat.append("MS SubClass")

skewed_columns = [
 'Lot Frontage',
 'Lot Area',
 'Mas Vnr Area',
 'BsmtFin SF 1',
 'BsmtFin SF 2',
 'Total Bsmt SF',
 '1st Flr SF',
 'Low Qual Fin SF',
 'Gr Liv Area',
 'Bsmt Half Bath',
 'Kitchen AbvGr',
 'Wood Deck SF',
 'Open Porch SF',
 'Enclosed Porch',
 '3Ssn Porch',
 'Screen Porch',
 'Pool Area',
 'Misc Val']

def apply_log(X):
  copy = X.copy()
  copy[skewed_columns] = copy[skewed_columns].apply(np.log1p)
  return copy

def create_features(X):
  copy = X.copy()
  copy['Tot Liv Area'] = copy['Total Bsmt SF'] + copy['Gr Liv Area']
  copy['Tot Rooms'] = copy['TotRms AbvGrd'] + copy['Bsmt Half Bath'] + copy['Bsmt Full Bath']
  copy['HouseAge'] = copy['Yr Sold'] - copy['Year Built']
  copy['Yrs Since Remod'] = copy['Yr Sold'] - copy['Year Remod/Add']
  return copy

eng_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('engineer', FunctionTransformer(func=create_features)),
    ('log1p', FunctionTransformer(func=apply_log)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value="missing")),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', eng_transformer, num),
    ('cat', categorical_transformer, cat)
])

preprocessor.set_output(transform="pandas")