import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

class DataProcessor:
    def __init__(self):
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        path = os.getcwd()
        self.train = pd.read_csv(os.path.join(path, 'data/train.csv'))
        self.val = pd.read_csv(os.path.join(path, 'data/val.csv'))

    def preprocess_data(self):
        self.train['people'] = self.train['rate_1'] + self.train['rate_2'] + self.train['rate_3']
        self.train = self.train[self.train['family_type'] != 1]
        self.train['size'] = self.train['size'].str.extract('(\d+\.?\d*)').astype(float)
        self.train.drop(columns=['application_type', 'name', 'address', 'family_type'], inplace=True)
        self.train['rate_1_ratio'] = self.train['rate_1'] / self.train['people']
        self.train['rate_2_ratio'] = self.train['rate_2'] / self.train['people']
        self.train['rate_3_ratio'] = self.train['rate_3'] / self.train['people']

        self.val = self.val[self.val['family_type'] != 1]
        self.val['size'] = self.val['size'].str.extract('(\d+\.?\d*)').astype(float)
        self.val.drop(columns=['application_type', 'name', 'address', 'family_type'], inplace=True)
        self.val['rate_1_ratio'] = self.val['rate_1'] / self.val['people']
        self.val['rate_2_ratio'] = self.val['rate_2'] / self.val['people']
        self.val['rate_3_ratio'] = self.val['rate_3'] / self.val['people']

        x = self.train.drop(columns=['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio'])
        y = self.train[['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio']]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.2)

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train_model(self):
        param = {
            'max_depth': range(1, 10),
            'n_estimators': range(1, 100, 10)
        }

        kfold = KFold(n_splits=5, shuffle=True)
        self.model = GridSearchCV(
            RandomForestRegressor(),
            param_grid=param,
            cv=kfold,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)

    def get_best_model(self):
        return self.model.best_estimator_

class ModelEvaluator:
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def evaluate(self):
        y_val_pred = self.model.predict(self.X_val)

        mse = mean_squared_error(self.y_val, y_val_pred)
        mae = mean_absolute_error(self.y_val, y_val_pred)
        r2 = r2_score(self.y_val, y_val_pred)

        print(f"Validation MSE: {mse:.5f}")
        print(f"Validation MAE: {mae:.5f}")
        print(f"Validation R2 Score: {r2:.5f}")

        return mse, mae, r2

class Predictor:
    def __init__(self, model, train_features, val_data):
        self.model = model
        self.train_features = train_features
        self.val_data = val_data

    def predict(self):
        val_x = self.val_data[self.train_features]
        pred_ratios = self.model.predict(val_x)

        pred_ratios_df = pd.DataFrame(
            pred_ratios, 
            columns=['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio']
        )

        self.val_data = self.val_data.reset_index(drop=True)
        self.val_data[['rate_1_ratio', 'rate_2_ratio', 'rate_3_ratio']] = pred_ratios_df

        self.val_data['rate_1'] = self.val_data['rate_1_ratio'] * self.val_data['people']
        self.val_data['rate_2'] = self.val_data['rate_2_ratio'] * self.val_data['people']
        self.val_data['rate_3'] = self.val_data['rate_3_ratio'] * self.val_data['people']

        print(self.val_data[['people', 'rate_1', 'rate_2', 'rate_3']])

if __name__ == "__main__":
    data_processor = DataProcessor()

    model_trainer = ModelTrainer(data_processor.X_train, data_processor.y_train)
    model_trainer.train_model()
    best_model = model_trainer.get_best_model()

    evaluator = ModelEvaluator(best_model, data_processor.X_val, data_processor.y_val)
    evaluator.evaluate()

    predictor = Predictor(best_model, data_processor.X_train.columns, data_processor.val)
    predictor.predict()