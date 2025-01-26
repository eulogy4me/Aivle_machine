import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

path = os.getcwd()

df = pd.read_csv(path + "/data/data.csv")
# 주소를 ~구 및 ~로,~길 만 남긴다.
df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
# 공급유형을 숫자로만
df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
df.drop(columns=['Address','Latitude','Longitude','Infra_score'], inplace=True)

le_gu = LabelEncoder()
le_ro = LabelEncoder()

df['gu'] = le_gu.fit_transform(df['gu'])
df['ro'] = le_ro.fit_transform(df['ro'])
df = pd.get_dummies(data=df.drop(columns=['gu','ro']))

X = df.drop(columns=[])
y = df['']