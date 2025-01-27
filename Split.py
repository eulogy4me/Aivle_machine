from models import Spliter
from sklearn.model_selection import train_test_split
import os
import joblib

def prepare(path):
    param_grid = {
        'max_depth': range(1, 51, 1),
        'n_estimators': range(1, 301, 1),
        'min_samples_split': range(2, 301, 1),
        'min_samples_leaf': range(1, 11)
    }
    
    data = path + "/data/data.csv"
    trainer = Spliter.ModelTrainer()
    X,y = trainer.preprocess_data(data)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    trainer.train_model(X_train,y_train,"RSCV",param_grid)
    trainer.evaluate_model(X_valid,y_valid)
    return trainer.split(X)

if __name__ == "__main__":
    path = os.getcwd()
    df = prepare(path)
    df.to_csv(path + "/data/type1.csv")