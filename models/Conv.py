import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class ModelTrainer:
    def __init__(self, random_state=-1):
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
        
        df.drop(
            columns=[
                'Address', 'Latitude', 'Longitude', 'Infra_score',
                'Cutline_rate','Cutline_score', 'Supply_type',
                'Applicant_type', 'Units','Gender','Shared','Year','Quarter',
                'Applied_type','people','Rate1','Rate2','Rate3'
            ],
            inplace=True
        )
        
        return df

    def train_model(self, X, y, search_method, param_grid):



        self.model.fit(X, y)
        
        print(f"Best Parameters: {self.model.best_params_}")
        print(f"Best Score (RMSE): {np.sqrt(-self.model.best_score_)}")

        self.model = self.model.best_estimator_

    def evaluate_model(self, X_valid, y_valid):
        predictions = self.model.predict(X_valid)
