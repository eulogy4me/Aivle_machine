from sklearn.metrics import mean_absolute_error , r2_score
from catboost import CatBoostRegressor, cv

class Model:
    def __init__(self):
        self.model = None
        self.result = None
        
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        model=CatBoostRegressor()
        self.model = model.load_model(path)

    def train(self, X, y, param_grid):                  
        self.model = CatBoostRegressor(
            loss_function='MultiRMSE',
            verbose=100,
            task_type='CPU'
        )
        
        self.result = self.model.grid_search(
            param_grid,
            X=X,
            y=y,
            cv=5,
            shuffle=True
        )
        
        print("Params : ", self.result['params'])
        # print("Score  : ", self.result['cv_results'])

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        try:
            print('Score MAE:', mean_absolute_error(y_test, y_pred))
            print('Score R2:', r2_score(y_test, y_pred))
        except:
            pass
        return y_pred