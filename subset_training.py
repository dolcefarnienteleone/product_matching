import pandas as pd

df = pd.read_csv('trainset_df_t10.csv')

df_P = df[df['label'] == 1]
df_N = df[df['label'] == 0]
df_P_5per = df_P.sample(frac=0.05, random_state = 31)
df_N_5per = df_N.sample(frac=0.05, random_state = 31)
frames = [df_P_5per, df_N_5per]
df_train = pd.concat(frames)
df_train = df_train.sample(frac=1).reset_index(drop=True)

df_train.to_csv('trainset_df_t13.csv', header=True, index=False)