import pandas as pd
import numpy as np
import time

np.random.seed(42)
import random

random.seed(42)

corpusj = pd.read_json('computers_train_xlarge.json', lines=True, chunksize=1)

i = 0 #0
cor_sent = []
t0 = time.time()
for chunk in corpusj:

    id_l = str(chunk.values[0][0])
    category_l = str(chunk.values[0][1])
    title_l = str(chunk.values[0][18])
    if title_l == 'nan':
        title_l = ''
    else:
        title_l = title_l
    brand_l = str(chunk.values[0][8])
    if brand_l == 'nan':
        brand_l = ''
    else:
        brand_l = brand_l
    description_l = str(chunk.values[0][10])
    if description_l == 'nan':
        description_l = ''
    else:
        description_l = description_l
    pairs_l = str(chunk.values[0][12])
    if pairs_l == 'nan':
        pairs_l = ''
    else:
        pairs_l = pairs_l
        #pairs_l = re.sub('[%s]' % re.escape(string.punctuation), '', pairs_l)
    id_r = str(chunk.values[0][3])
    category_r = str(chunk.values[0][4])
    title_r = str(chunk.values[0][19])
    if title_r == 'nan':
        title_r = ''
    else:
        title_r = title_r
    brand_r = str(chunk.values[0][9])
    if brand_r == 'nan':
        brand_r = ''
    else:
        brand_r = brand_r
    description_r = str(chunk.values[0][11])
    if description_r == 'nan':
        description_r = ''
    else:
        description_r = description_r
    pairs_r = str(chunk.values[0][13])
    if pairs_r == 'nan':
        pairs_r = ''
    else:
        pairs_r = pairs_r
        #pairs_r = re.sub('[%s]' % re.escape(string.punctuation), '', pairs_r)

    cor_sent.append(f'{id_l} {category_l}\n{id_l} {title_l}\n{id_l} {brand_l}\n'
                    f'{id_l} {description_l}\n{id_l} {pairs_l}\n'
                    f'{id_r} {category_r}\n{id_r} {title_r}\n{id_r} {brand_r}\n'
                    f'{id_r} {description_r}\n{id_r} {pairs_r}\n')

    #print("\n")

    #if i > 10:
        #break
    #else:
        #i = i + 1
    i = i + 1
print(i)
t1 = time.time()
corfile = open("train_corpus_v5.txt", "w")
corfile.writelines(cor_sent)
corfile.close()
t2 = time.time()
print(f'read file spend time {t1-t0}')
print(f'write into txt file spend time {t2-t1}')