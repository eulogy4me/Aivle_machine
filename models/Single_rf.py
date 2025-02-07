import numpy as np
import joblib
import logging
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from time import time

class Model:
    def __init__(self):
        self.model = None
        self.result = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def save(self, path):
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load(self, path):
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def train(self, X_train, y_train, param_grid):
        start_time = time()

        self.model = RandomForestRegressor(n_jobs=-1)

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.result = grid_search.best_params_

        end_time = time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Best parameters: {self.result}")

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"R2: {r2:.4f}")

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")

        return y_pred