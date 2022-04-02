import pandas as pd
import numpy as np

df = pd.read_csv('offline/transactions_train.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

train.to_csv('offline/split/train.csv')
test.to_csv('offline/split/test.csv')