import pandas as pd
import numpy as np 

df = pd.read_csv('./train.csv')
groups = df.groupby('Id')
ng = groups.ngroup()
# e = np.eye(len(groups))
# for i in ng.values:
#     e[i]
# print(ng)
df['class'] = ng
df.to_csv('./train_labled.csv')


