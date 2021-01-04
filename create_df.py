import pandas as pd
import numpy as np

np.random.seed(42)
import random

random.seed(42)

#data = pd.read_json('computers_train_xlarge.json', lines=True, chunksize=1)
#data = pd.read_json('computers_validation.json', lines=True, chunksize=1)
data = pd.read_json('task1_testset_1500_with_labels.json', lines=True, chunksize=1)

i = 0 #0
left_id = []
left_category = []
right_id = []
right_category = []
label = []
left_brand = []
right_brand = []
left_keyValuePairs = []
right_keyValuePairs = []
left_title = []
right_title = []

for chunk in data:

    #label.append(chunk.values[0][6])
    label.append(chunk.values[0][18])
    left_id.append(chunk.values[0][0])
    #left_category.append(chunk.values[0][1])
    left_category.append(chunk.values[0][2])
    #left_title.append(chunk.values[0][18])
    left_title.append(chunk.values[0][3])
    #left_brand.append(chunk.values[0][8])
    left_brand.append(chunk.values[0][5])
    #left_keyValuePairs.append(chunk.values[0][12])
    left_keyValuePairs.append(chunk.values[0][7])
    #right_id.append(chunk.values[0][3])
    #right_category.append(chunk.values[0][4])
    #right_title.append(chunk.values[0][19])
    #right_brand.append(chunk.values[0][9])
    #right_keyValuePairs.append(chunk.values[0][13])
    right_id.append(chunk.values[0][9])
    right_category.append(chunk.values[0][11])
    right_title.append(chunk.values[0][12])
    right_brand.append(chunk.values[0][14])
    right_keyValuePairs.append(chunk.values[0][16])

    #if i > 100:
        #break
    #else:
        #i = i + 1
    i = i+1

df = pd.DataFrame(list(zip(label, left_id, left_category, left_title, left_brand, left_keyValuePairs,
                           right_id, right_category, right_title, right_brand, right_keyValuePairs)),
     columns = ['label', 'left_id', 'left_category', 'left_title', 'left_brand', 'left_keyValuePairs',
                'right_id', 'right_category', 'right_title', 'right_brand', 'right_keyValuePairs'])

#df.to_csv('validset_df_t10.csv', header=True)
df.to_csv('testset_df_t10.csv',header=True)