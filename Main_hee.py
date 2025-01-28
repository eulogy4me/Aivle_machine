from models import hee
import os
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    RANDOM_STATE = 42
    param_grid = {
        'max_depth': range(1, 21),
        'n_estimators': range(50, 301, 50),
        'min_samples_split': range(2, 21, 1),
        'min_samples_leaf': range(1, 11)
    }

    path = os.getcwd()
    file_path = os.path.join(path, "data/data.csv")

    trainer = hee.ModelTrainer(random_state=RANDOM_STATE)
    df = trainer.preprocess_data(file_path)

    X = df.drop(columns=['Qty'])
    y = df['Qty']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("RandomizedSearchCV Results:")
    trainer.train_model(X_train, y_train, "RandomizedSearchCV", param_grid)
    trainer.save(path + '/hee.pkl')
    trainer.load(path + '/hee.pkl')
    trainer.evaluate_model(X_valid, y_valid)