import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.model = None
        
    def save(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved {filepath}")
        else:
            print("No model to save")
            
    def load(self, filepath):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
        df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
        qty = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

        df.drop(
            columns=[
                'Address', 'Latitude', 'Longitude', 'Infra_score',
                'Cutline_rate','Cutline_score','Units',
                'Gender','Shared','Year','Quarter','Applied_type','Counts_daiso',
                'Counts_supermarket','Counts_laundry','Counts_pharmacy','Counts_cafe',
                'Counts_convstore',
            ],
            inplace=True
        )
        
        df = pd.get_dummies(data=df)
        df['Qty'] = qty

        return df

    def train_model(self, X, y, search_method, param_grid):
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        if search_method == "RandomizedSearchCV":
            self.model = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                n_iter=100,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )
        elif search_method == "HalvingGridSearchCV":
            self.model = HalvingGridSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_grid=param_grid,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )

        self.model.fit(X, y)
        
        print(f"Best Parameters: {self.model.best_params_}")
        print(f"Best Score (RMSE): {np.sqrt(-self.model.best_score_)}")

        self.model = self.model.best_estimator_

    def evaluate_model(self, X_valid, y_valid):
        predictions = self.model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        r2 = r2_score(y_valid, predictions)

        print(f"Validation RMSE: {rmse}")
        print(f"Validation R²: {r2}")
