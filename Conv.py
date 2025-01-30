from models import hee, Conv
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    path = os.getcwd()
    file_path = path + "/data/data.csv"
    trainer = Conv.ModelTrainer(random_state=42, n_clusters=5)
    df = pd.read_csv(file_path)
    
    features = trainer.train_model(file_path)
    
    convenience_scores = trainer.calculate_convenience_score(features)
    print("Convenience scores calculated:")
    print(convenience_scores)
    
    df['ConvScore'] = convenience_scores
    
    print(df.head())
    
    trainer.save("ConvScore.pkl")
    
    df.to_csv(path + "/data/data_ConvScore.csv", index=False)