from preprocessing import X, y, preprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            subsample=0.8,
            n_estimators=200,
            min_samples_leaf=1,
            min_samples_split=10,
            max_depth=4,
            learning_rate=0.05
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

model.fit(X_train, y_train)

joblib.dump(model, '../model/model.joblib')

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"mse: {mse}")
print(f"R2: {r2}")