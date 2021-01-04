import pandas as pd
import numpy as np

df = pd.read_csv('testset_df_t10.csv')
###separate the df with positive and negative samples to make sure the ratio in each subset is the same
df_P = df[df['label'] == 1]
df_N = df[df['label'] == 0]
### separate the testset with the intended split, do a random shuffle before the split
partition_num = 3
df_P_1, df_P_2, df_P_3 = np.split(df_P.sample(frac=1, random_state = 31), partition_num)
df_N_1, df_N_2, df_N_3 = np.split(df_N.sample(frac=1, random_state = 31), partition_num)

### combine the positive and negative samples for new subset of testsets
frames1 = [df_P_1, df_N_1]
df_train_1 = pd.concat(frames1)
df_train_1 = df_train_1.sample(frac=1, random_state = 31).reset_index(drop=True)
df_train_1.to_csv('testset_df_t13_1.csv', header=True, index=False)

frames2 = [df_P_2, df_N_2]
df_train_2 = pd.concat(frames2)
df_train_2 = df_train_2.sample(frac=1, random_state = 31).reset_index(drop=True)
df_train_2.to_csv('testset_df_t13_2.csv', header=True, index=False)

frames3 = [df_P_3, df_N_3]
df_train_3 = pd.concat(frames3)
df_train_3 = df_train_3.sample(frac=1, random_state = 31).reset_index(drop=True)
df_train_3.to_csv('testset_df_t13_3.csv', header=True, index=False)
