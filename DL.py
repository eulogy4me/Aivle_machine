from models.Struct import Model
import os
import pandas as pd

if __name__ == "__main__":
    path = os.getcwd()
    dataset_path = path + "/datasets"
    models_path = path + "/pkl/struct.pth"
    csv_path = path + "/rslt/struct.csv"

    trainer = Model(dataset_path)
    trainer.train(models_path)
    trainer.evaluate(models_path,csv_path)

    df = pd.read_csv(csv_path)
    min_score = df["Predicted Score"].min()
    max_score = df["Predicted Score"].max()
    df["Scaled Score"] = ((df["Predicted Score"] - min_score) / (max_score - min_score)) * (5 - 1) + 1
    print(df[["File Name", "Predicted Score", "Scaled Score"]])
    df.drop(columns=['True Score'])
    df.to_csv("scaled_scores.csv", index=False)
