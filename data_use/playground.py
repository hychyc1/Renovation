import numpy as np
import pandas as pd

df = pd.read_csv('village_with_col.csv')
df = df.astype({'assign_row': 'int32', 'assign_col': 'int32'})
print(df.values)