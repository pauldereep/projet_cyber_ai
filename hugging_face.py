import pandas as pd

df = pd.read_json("hf://datasets/pauldereep/test/modified_dataset.csv")

print(df.head())