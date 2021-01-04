import re
import string
import gensim
import nltk
import numpy as np
import pandas as pd
import logging
import time
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from scipy.spatial import distance
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, \
    remove_special_character, check_spelling, remove_stopword, tokenize_word, stem_word

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import warnings

warnings.filterwarnings(action='ignore')

### functions

# train_test split
def split_train_valid(df, test_size=0.3, shuffle_state=True):
    X_train, X_valid, Y_train, Y_valid = train_test_split(df.iloc[:, 1:],
                                                        df['label'],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    print(f'Train set value counts:')
    print(Y_train.value_counts())
    print(f'Valid set value counts:')
    print(Y_valid.value_counts())
    print(X_train.head())
    return X_train, X_valid, Y_train, Y_valid

### when preprocess is needed to test
#preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, remove_special_character,
                        #check_spelling, remove_stopword, lemmatize_word]
# single word column tokenizer
def sinword_tokenize(text):
    text = str(text)
    text = re.sub('@en', '', text)
    text = re.sub('@es', '', text)
    text = re.sub('@fr', '', text)
    #text = preprocess_text(text, preprocess_functions)
    tokens = []
    for word in sent_tokenize(text):
        tokens.append(word)
    return tokens

# sentence column tokenizer

def sent_tokenize(text):
    text = str(text)
    text = re.sub('@en', '', text)
    text = re.sub('@es', '', text)
    text = re.sub('@fr', '', text)
    #text = preprocess_text(text, preprocess_functions)
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens

# get average from word or sentence vector

def word_averaging(model, words):
    mean = []
    for w in words:
        if isinstance(w, np.ndarray):
            mean.append(w)
        elif w in model.wv.vocab:
            mean.append(model.wv.get_vector(w))

    if not mean:

        return np.zeros(model.wv.vector_size,)

    mean = np.array(mean).mean(axis=0).astype(np.float32)
    return mean

def  word_averaging_list(model, text_list):
    return np.vstack([word_averaging(model, text) for text in text_list])

# get final product vector

def get_product_vec(model, a, b, c, d, e):
    a_avg = word_averaging_list(model, a)
    b_avg = word_averaging_list(model, b)
    c_avg = word_averaging_list(model, c)
    d_avg = word_averaging_list(model, d)
    e_avg = word_averaging_list(model, e)
    sum_all = np.add(a_avg, b_avg)
    sum_all = np.add(sum_all, c_avg)
    sum_all = np.add(sum_all, d_avg)
    sum_all = np.add(sum_all, e_avg)
    return sum_all

# Function for comparison of cross-validation performance
from sklearn.model_selection import cross_val_score
def cv_comparison(models, X, y, cv):
    cv_report = pd.DataFrame()
    accs = []
    prcs = []
    rcs = []
    f1ss = []
    for model in models:
        acc = np.round(cross_val_score(model, X, y, scoring='accuracy', cv=cv), 4)*100
        accs.append(acc)
        acc_avg = round(acc.mean(), 4)
        prc = np.round(cross_val_score(model, X, y, scoring='precision', cv=cv), 4)*100
        prcs.append(prc)
        prc_avg = round(prc.mean(), 4)
        rc = np.round(cross_val_score(model, X, y, scoring='recall', cv=cv), 4)*100
        rcs.append(rc)
        rc_avg = round(rc.mean(), 4)
        f1s = np.round(cross_val_score(model, X, y, scoring='f1', cv=cv), 4)*100
        f1ss.append(f1s)
        f1s_avg = round(f1s.mean(), 4)
        cv_report[str(model)] = [acc_avg, prc_avg, rc_avg, f1s_avg]
    cv_report.index = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    return cv_report, accs, prcs, rcs, f1ss

### load df
df_train = pd.read_csv('trainset_df_t10.csv')
df_train = df_train.drop(['Unnamed: 0'], axis=1)
df_valid = pd.read_csv('validset_df_t10.csv')
df_valid = df_valid.drop(['Unnamed: 0'], axis=1)
#adding subset of testing set to do experiment
#sub_test = pd.read_csv('testset_df_t13_1.csv')
#sub_test = sub_test.drop(['Unnamed: 0'], axis=1)

df_test = pd.read_csv('testset_df_t10.csv')
df_test = df_test.drop(['Unnamed: 0'], axis=1)

###depending on the experiment setting, changing with different combination of datasets
#frames = [df_train, df_valid]
#frames = [df_train, df_valid, df_test]
#frames = [df_train, df_valid, sub_test]
#df = pd.concat(frames, ignore_index=True)

### load word2vec model
w2vbw = gensim.models.Word2Vec.load('w2v_cbow_v3.model')

### Call the train_valid_split
#X_train, X_valid, Y_train, Y_valid = split_train_valid(df)
#X_train, X_valid, Y_train, Y_valid = split_train_valid(df_train)

X_train = df_train.iloc[:, 1:]
Y_train = df_train['label']

X_valid = df_valid.iloc[:, 1:]
Y_valid = df_valid['label']

### Preprocess text to vector
t1 = time.time()
train_Lid_tok = X_train.apply(lambda r: sinword_tokenize(r['left_id']), axis=1).values
train_Lcat_tok = X_train.apply(lambda r: sinword_tokenize(r['left_category']), axis=1).values
train_Lbrand_tok = X_train.apply(lambda r: sent_tokenize(r['left_brand']), axis=1).values
train_Ltitle_tok = X_train.apply(lambda r: sent_tokenize(r['left_title']), axis=1).values
train_Lkey_tok = X_train.apply(lambda r: sent_tokenize(r['left_keyValuePairs']), axis=1).values
train_Rid_tok = X_train.apply(lambda r: sinword_tokenize(r['right_id']), axis=1).values
train_Rcat_tok = X_train.apply(lambda r: sinword_tokenize(r['right_category']), axis=1).values
train_Rbrand_tok = X_train.apply(lambda r: sent_tokenize(r['right_brand']), axis=1).values
train_Rtitle_tok = X_train.apply(lambda r: sent_tokenize(r['right_title']), axis=1).values
train_Rkey_tok = X_train.apply(lambda r: sent_tokenize(r['right_keyValuePairs']), axis=1).values

valid_Lid_tok = X_valid.apply(lambda r: sinword_tokenize(r['left_id']), axis=1).values
valid_Lcat_tok = X_valid.apply(lambda r: sinword_tokenize(r['left_category']), axis=1).values
valid_Lbrand_tok = X_valid.apply(lambda r: sent_tokenize(r['left_brand']), axis=1).values
valid_Ltitle_tok = X_valid.apply(lambda r: sent_tokenize(r['left_title']), axis=1).values
valid_Lkey_tok = X_valid.apply(lambda r: sent_tokenize(r['left_keyValuePairs']), axis=1).values
valid_Rid_tok = X_valid.apply(lambda r: sinword_tokenize(r['right_id']), axis=1).values
valid_Rcat_tok = X_valid.apply(lambda r: sinword_tokenize(r['right_category']), axis=1).values
valid_Rbrand_tok = X_valid.apply(lambda r: sent_tokenize(r['right_brand']), axis=1).values
valid_Rtitle_tok = X_valid.apply(lambda r: sent_tokenize(r['right_title']), axis=1).values
valid_Rkey_tok = X_valid.apply(lambda r: sent_tokenize(r['right_keyValuePairs']), axis=1).values

X_test = df_test.iloc[:, 1:]
Y_test = df_test['label']

test_Lid_tok = X_test.apply(lambda r: sinword_tokenize(r['left_id']), axis=1).values
test_Lcat_tok = X_test.apply(lambda r: sinword_tokenize(r['left_category']), axis=1).values
test_Lbrand_tok = X_test.apply(lambda r: sent_tokenize(r['left_brand']), axis=1).values
test_Ltitle_tok = X_test.apply(lambda r: sent_tokenize(r['left_title']), axis=1).values
test_Lkey_tok = X_test.apply(lambda r: sent_tokenize(r['left_keyValuePairs']), axis=1).values
test_Rid_tok = X_test.apply(lambda r: sinword_tokenize(r['right_id']), axis=1).values
test_Rcat_tok = X_test.apply(lambda r: sinword_tokenize(r['right_category']), axis=1).values
test_Rbrand_tok = X_test.apply(lambda r: sent_tokenize(r['right_brand']), axis=1).values
test_Rtitle_tok = X_test.apply(lambda r: sent_tokenize(r['right_title']), axis=1).values
test_Rkey_tok = X_test.apply(lambda r: sent_tokenize(r['right_keyValuePairs']), axis=1).values

## concatenate all features

train_Lid_vec = word_averaging_list(w2vbw, train_Lid_tok)
train_Lcat_vec = word_averaging_list(w2vbw, train_Lcat_tok)
train_Lbrand_vec = word_averaging_list(w2vbw, train_Lbrand_tok)
train_Ltitle_vec = word_averaging_list(w2vbw, train_Ltitle_tok)
train_Lkey_vec = word_averaging_list(w2vbw, train_Lkey_tok)
train_Rid_vec = word_averaging_list(w2vbw, train_Rid_tok)
train_Rcat_vec = word_averaging_list(w2vbw, train_Rcat_tok)
train_Rbrand_vec = word_averaging_list(w2vbw, train_Rbrand_tok)
train_Rtitle_vec = word_averaging_list(w2vbw, train_Rtitle_tok)
train_Rkey_vec = word_averaging_list(w2vbw, train_Rkey_tok)

valid_Lid_vec = word_averaging_list(w2vbw, valid_Lid_tok)
valid_Lcat_vec = word_averaging_list(w2vbw, valid_Lcat_tok)
valid_Lbrand_vec = word_averaging_list(w2vbw, valid_Lbrand_tok)
valid_Ltitle_vec = word_averaging_list(w2vbw, valid_Ltitle_tok)
valid_Lkey_vec = word_averaging_list(w2vbw, valid_Lkey_tok)
valid_Rid_vec = word_averaging_list(w2vbw, valid_Rid_tok)
valid_Rcat_vec = word_averaging_list(w2vbw, valid_Rcat_tok)
valid_Rbrand_vec = word_averaging_list(w2vbw, valid_Rbrand_tok)
valid_Rtitle_vec = word_averaging_list(w2vbw, valid_Rtitle_tok)
valid_Rkey_vec = word_averaging_list(w2vbw, valid_Rkey_tok)

test_Lid_vec = word_averaging_list(w2vbw, test_Lid_tok)
test_Lcat_vec = word_averaging_list(w2vbw, test_Lcat_tok)
test_Lbrand_vec = word_averaging_list(w2vbw, test_Lbrand_tok)
test_Ltitle_vec = word_averaging_list(w2vbw, test_Ltitle_tok)
test_Lkey_vec = word_averaging_list(w2vbw, test_Lkey_tok)
test_Rid_vec = word_averaging_list(w2vbw, test_Rid_tok)
test_Rcat_vec = word_averaging_list(w2vbw, test_Rcat_tok)
test_Rbrand_vec = word_averaging_list(w2vbw, test_Rbrand_tok)
test_Rtitle_vec = word_averaging_list(w2vbw, test_Rtitle_tok)
test_Rkey_vec = word_averaging_list(w2vbw, test_Rkey_tok)

## get one vec for product by sum
#train_left_p = get_product_vec(w2vbw, train_Lid_tok, train_Lcat_tok, train_Lbrand_tok, train_Ltitle_tok, train_Lkey_tok)
#train_right_p = get_product_vec(w2vbw, train_Rid_tok, train_Rcat_tok, train_Rbrand_tok, train_Rtitle_tok, train_Rkey_tok)
#valid_left_p = get_product_vec(w2vbw, valid_Lid_tok, valid_Lcat_tok, valid_Lbrand_tok, valid_Ltitle_tok, valid_Lkey_tok)
#valid_right_p = get_product_vec(w2vbw, valid_Rid_tok, valid_Rcat_tok, valid_Rbrand_tok, valid_Rtitle_tok, valid_Rkey_tok)
#test_left_p = get_product_vec(w2vbw, test_Lid_tok, test_Lcat_tok, test_Lbrand_tok, test_Ltitle_tok, test_Lkey_tok)
#test_right_p = get_product_vec(w2vbw, test_Rid_tok, test_Rcat_tok, test_Rbrand_tok, test_Rtitle_tok, test_Rkey_tok)

#Final_train = np.subtract(train_left_p, train_right_p)
#Final_train = np.concatenate((train_left_p, train_right_p), axis=1)
Final_train = np.concatenate((train_Lid_vec, train_Lcat_vec, train_Lbrand_vec, train_Ltitle_vec, train_Lkey_vec,
                              train_Rid_vec, train_Rcat_vec, train_Rbrand_vec, train_Rtitle_vec, train_Rkey_vec), axis=1)

#Final_valid = np.subtract(valid_left_p, valid_right_p)
#Final_valid = np.concatenate((valid_left_p, valid_right_p), axis=1)
Final_valid = np.concatenate((valid_Lid_vec, valid_Lcat_vec, valid_Lbrand_vec, valid_Ltitle_vec, valid_Lkey_vec,
                              valid_Rid_vec, valid_Rcat_vec, valid_Rbrand_vec, valid_Rtitle_vec, valid_Rkey_vec), axis=1)

#Final_test = np.subtract(test_left_p, test_right_p)
#Final_test = np.concatenate((test_left_p, test_right_p), axis=1)
Final_test = np.concatenate((test_Lid_vec, test_Lcat_vec, test_Lbrand_vec, test_Ltitle_vec, test_Lkey_vec,
                              test_Rid_vec, test_Rcat_vec, test_Rbrand_vec, test_Rtitle_vec, test_Rkey_vec), axis=1)

t2 = time.time()
print(f'time spends on pre-processing: {t2-t1}')

### start ML classifier
print(f'start training')

tags = ['Non-Matching', 'Matching']

### Logistic regression
from sklearn.linear_model import LogisticRegression
t1 = time.time()
LR = LogisticRegression()
LR = LR.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = LR.predict(Final_test)

print(f'LR validation accuracy: {round(LR.score(Final_valid, Y_valid), 4)}')
print(f'LR testing accuracy: {round(LR.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred, target_names=tags))
print(f'LR training time: {t2-t1}')

### Random Forest
from sklearn.ensemble import RandomForestClassifier
t1 = time.time()
RF = RandomForestClassifier()
RF = RF.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = RF.predict(Final_test)
print(f'RF validation accuracy: {round(RF.score(Final_valid, Y_valid), 4)}')
print(f'RF testing accuracy: {round(RF.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred, target_names=tags))
print(f'RF training time: {t2-t1}')

### NN
from sklearn.neural_network import MLPClassifier
t1 = time.time()
NN = MLPClassifier()
NN = NN.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = NN.predict(Final_test)
print(f'NN validation accuracy: {round(NN.score(Final_valid, Y_valid), 4)}')
print(f'NN testing accuracy: {round(NN.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred, target_names=tags))
print(f'NN training time: {t2-t1}')

print(f'start doing cv comparison table')
t1 = time.time()
LR = LogisticRegression()
RF = RandomForestClassifier()
NN = MLPClassifier()
models = [LR, RF, NN]
comp, accs, prcs, rcs, f1ss = cv_comparison(models, Final_train, Y_train, 3) #5
t2 = time.time()
print(f'comparison table is finished, time cost:{t2-t1}')
print(comp)
