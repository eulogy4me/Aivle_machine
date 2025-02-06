import models
import os
import pandas as pd

if __name__ == "__main__":
    path = os.getcwd()
    data = pd.read_csv(path + "/data/data.csv")
    Input = pd.read_csv(path + "/data/input.csv")
    
    PEOPLE = models.People.ModelTrainer()
    SPLITER = models.Spliter.ModelTrainer()
    QTY = models.Qty.ModelTrainer()

    PEOPLE.load(path + "/pkl/people.cbm")
    SPLITER.load(path + "/pkl.split.cbm")
    QTY.load(path + "/pkl/qty.cbm")