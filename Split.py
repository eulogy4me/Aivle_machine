from models import Spliter
from sklearn.model_selection import train_test_split
import os
import joblib

if __name__ == "__main__":
    RANDOM_STATE = 42
    path = os.getcwd() + "data"
    param_grid = {
        'max_depth': range(1, 21),
        'n_estimators': range(50, 301, 50),
        'min_samples_split': range(2, 21, 1),
        'min_samples_leaf': range(1, 11)
    }
    
    data = os.path.join(path, "/data.csv")
    trainer = Spliter.ModelTrainer(random_state=RANDOM_STATE)
    X,y = trainer.preprocess_data(data)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    trainer.train_model(X_train,y_train,"RSCV",param_grid)
    trainer.evaluate_model(X_valid,y_valid)
    df = trainer.split(X)
    df.to_csv(os.path.join(path, "/type1.csv"))