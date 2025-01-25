import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error


path = os.getcwd()
train = pd.read_csv(path + '/data/train.csv')
val = pd.read_csv(path + '/data/val.csv')

train['people'] = train['rate_1'] + train['rate_2'] + train['rate_3']
train = train[train['family_type'] != 1]
train['size'] = train['size'].str.extract('(\d+\.?\d*)').astype(float)
train.drop(columns=['application_type','name','address','family_type'], inplace=True)
train['rate_1_ratio'] = train['rate_1'] / train['people']
train['rate_2_ratio'] = train['rate_2'] / train['people']
train['rate_3_ratio'] = train['rate_3'] / train['people']

val = val[val['family_type'] != 1]
val['size'] = val['size'].str.extract('(\d+\.?\d*)').astype(float)
val.drop(columns=['application_type','name','address','family_type'], inplace=True)
val['rate_1_ratio'] = val['rate_1'] / val['people']
val['rate_2_ratio'] = val['rate_2'] / val['people']
val['rate_3_ratio'] = val['rate_3'] / val['people']

x = train.drop(columns=['rate_1_ratio','rate_2_ratio','rate_3_ratio'])
y = train[['rate_1_ratio','rate_2_ratio','rate_3_ratio']]

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)


param = {
    'max_depth': range(1, 10),
    'n_estimators': range(1, 100, 10)
}

kfold = KFold(
    n_splits=5,
    shuffle=True
)

model = GridSearchCV(
    RandomForestRegressor(),
    param_grid=param,
    cv=kfold,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1
)

model.fit(X_train, y_train)

y_val_pred = model.best_estimator_.predict(X_val)

# MSE (Validation MSE)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {mse:.5f}")

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Validation MAE: {mae:.5f}")

# R2 Score (결정 계수)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R2 Score: {r2:.5f}")

y_val_pred = model.best_estimator_.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {val_mse:.2f}")

train_features = X_train.columns
val_x = val[train_features]
pred_ratios = model.best_estimator_.predict(val_x)

pred_ratios_df = pd.DataFrame(
    pred_ratios, 
    columns=['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio']
)

val = val.reset_index(drop=True)
val[['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio']] = pred_ratios_df

val['rate_1'] = val['rate_1_ratio'] * val['people']
val['rate_2'] = val['rate_2_ratio'] * val['people']
val['rate_3'] = val['rate_3_ratio'] * val['people']

print(val[['people', 'rate_1', 'rate_2', 'rate_3']])