import numpy as np
import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, random_state=42, n_clusters=5):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        
    def save(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved at {filepath}")
        else:
            print("No model to save")
            
    def load(self, filepath):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        
        df.drop(
            columns=[
                'Name',
                'Address', 'Latitude', 'Longitude', 'Infra_score',
                'Cutline_rate','Cutline_score', 'Supply_type',
                'Applicant_type', 'Units','Gender','Shared','Year','Quarter',
                'Applied_type','people','Rate1','Rate2','Rate3'
            ],
            inplace=True
        )
        
        return df

    def train_model(self, filepath):
        df = self.preprocess_data(filepath)
        df.fillna(0, inplace=True)

        features = self.scaler.fit_transform(df.values)
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.model.fit(features)
        
        print("Model trained with KMeans clustering")
        return features

    def calculate_convenience_score(self, features):
        distances = self.model.transform(features)
        scores = 1 / (1 + distances.min(axis=1))
        return scores
