import pandas as pd
import os

path = os.getcwd()
people = pd.read_csv("/root/Aivle_machine/rslt/people.csv")
people.columns = people.columns.str.strip()

print(people.columns)

test = pd.read_csv(path + "/data/test.csv")

df = pd.DataFrame()
df['test_people'] = test['people']
df['Name'] = test['Name']
df['pred_people'] = people['people']

print(df)
