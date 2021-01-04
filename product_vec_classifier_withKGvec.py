import pandas as pd
import numpy as np
import re
import string

import time as time

### functions
import spacy
import en_core_web_sm
from kg.Lookup import DBpediaLookup
from kg.Lookup import WikidataAPI
# within get KG vector
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

import requests
from bs4 import BeautifulSoup
import json

# get KG vector
nlp = en_core_web_sm.load()
dbpedia = DBpediaLookup()
wikidata = WikidataAPI()

def spacy_ent_to_vec(text, kg, vec_size):
    text = str(text)
    text = re.sub('@en', '', text)
    text = re.sub('@es', '', text)
    text = re.sub('@fr', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    doc = nlp(text)
    ent_candidate = []
    for spacy_ent in doc.ents:
        ent_candidate.append(spacy_ent)
    for token in doc:
        if token.pos_ == 'NOUN':
            ent_candidate.append(token)
    kg = kg
    vec_size = vec_size
    empty_vec = np.zeros(vec_size)
    limit = 1
    mean = []
    for j in ent_candidate:
        query = j
        if query in query_entity_dict:
            ent = query_entity_dict.get(query)
            if ent in entity_vec_dict:
                ent_vec = entity_vec_dict.get(ent)
                # entities.append(entity_dict.get(query))
                mean.append(ent_vec)
            else:
                url = 'http://www.kgvec2go.org/rest/get-vector/' + kg + '/' + ent
                res = requests.get(url)
                if res.status_code == 54:
                    time.sleep(60)  # to avoid limit of calls, sleep 60s
                    res_post = requests.get(url)
                    if res_post.status_code == 200:
                        html = res_post.text
                        if html == '{}':
                            mean.append(empty_vec)
                            entity_vec_dict[ent] = empty_vec
                        else:
                            soup = BeautifulSoup(html, 'html.parser')
                            data = " ".join(re.split(r'[\n\t]+', soup.get_text()))
                            data = json.loads(data)
                            vec = np.array(data['vector'])
                            mean.append(vec)
                            entity_vec_dict[ent] = vec
                elif res.status_code == 200:
                    html = res.text
                    if html == '{}':
                        mean.append(empty_vec)
                        entity_vec_dict[ent] = empty_vec
                    else:
                        soup = BeautifulSoup(html, 'html.parser')
                        data = " ".join(re.split(r'[\n\t]+', soup.get_text()))
                        data = json.loads(data)
                        vec = np.array(data['vector'])
                        mean.append(vec)
                        entity_vec_dict[ent] = vec

                else:
                    mean.append(empty_vec)
                    entity_vec_dict[ent] = empty_vec
        else:
            entity = dbpedia.getKGEntities(query, limit)
            #entity = wikidata.getKGEntities(query, limit, "item")
            #entity = googleKG.getKGEntities(query, limit)
            if entity == []:
                mean.append(empty_vec)
                query_entity_dict[query] = None
                entity_vec_dict[query] = empty_vec
                # entity_dict[query] = None
            else:
                ent = remove_tags(entity[0])
                query_entity_dict[query] = ent
                # ent = entity[0]
                if ent in entity_vec_dict:
                    ent_vec = entity_vec_dict.get(ent)
                    # entities.append(entity_dict.get(query))
                    mean.append(ent_vec)
                else:
                    # try:
                    url = 'http://www.kgvec2go.org/rest/get-vector/' + kg + '/' + ent
                    res = requests.get(url)
                    if res.status_code == 54:
                        time.sleep(60)  # to avoid limit of calls, sleep 60s
                        res_post = requests.get(url)
                        if res_post.status_code == 200:
                            html = res_post.text
                            if html == '{}':
                                mean.append(empty_vec)
                                entity_vec_dict[ent] = empty_vec
                            else:
                                soup = BeautifulSoup(html, 'html.parser')
                                data = " ".join(re.split(r'[\n\t]+', soup.get_text()))
                                data = json.loads(data)
                                vec = np.array(data['vector'])
                                mean.append(vec)
                                entity_vec_dict[ent] = vec
                    elif res.status_code == 200:
                        html = res.text
                        if html == '{}':
                            mean.append(empty_vec)
                            entity_vec_dict[ent] = empty_vec
                        else:
                            soup = BeautifulSoup(html, 'html.parser')
                            data = " ".join(re.split(r'[\n\t]+', soup.get_text()))
                            data = json.loads(data)
                            vec = np.array(data['vector'])
                            mean.append(vec)
                            entity_vec_dict[ent] = vec

                    else:
                        mean.append(empty_vec)
                        entity_vec_dict[ent] = empty_vec
    else:
        mean.append(empty_vec)
        # mean = np.array(mean).mean(axis=0)
        # mean = empty_vec

    mean = np.array(mean).mean(axis=0)
    mean = np.vstack(mean)
    mean = mean.T

    return mean

import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# single word column tokenizer

def sinword_tokenize(text):
    text = str(text)
    text = re.sub('@en', '', text)
    text = re.sub('@es', '', text)
    text = re.sub('@fr', '', text)
    # text = preprocess_text(text, preprocess_functions)
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
    # text = preprocess_text(text, preprocess_functions)
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
        return np.zeros(model.wv.vector_size, )

    mean = np.array(mean).mean(axis=0).astype(np.float32)
    return mean

def  word_averaging_list(model, text_list):
    return np.vstack([word_averaging(model, text) for text in text_list])


# train_test split
from sklearn.model_selection import train_test_split

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

t00 = time.time()
### load df
#df_train = pd.read_csv('trainset_df_t10.csv')
#df_train = df_train.drop(['Unnamed: 0'], axis=1)
df_train = pd.read_csv('trainset_df_t13.csv')
df_train = df_train.drop(['Unnamed: 0'], axis=1)
df_valid = pd.read_csv('validset_df_t10.csv')
df_valid = df_valid.drop(['Unnamed: 0'], axis=1)
df_test = pd.read_csv('testset_df_t10.csv')
df_test = df_test.drop(['Unnamed: 0'], axis=1)

frames = [df_train, df_valid]
df = pd.concat(frames, ignore_index=True)

### Call the train_valid_split
X_train, X_valid, Y_train, Y_valid = split_train_valid(df)

### define test set
X_test = df_test.iloc[:, 1:]
Y_test = df_test['label']

### load word2vec model
w2vbw = gensim.models.Word2Vec.load('w2v_cbow_v3.model')

### Preprocess text to word vector by pre-trained w2v model
t1 = time.time()
#tokenization training set
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

#tokenization validation set
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

#tokenization testing set
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

#retrieve word vector from pre-trained model and do averaging for sentences_training set
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

#retrieve word vector from pre-trained model and do averaging for sentences_validation set

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

#retrieve word vector from pre-trained model and do averaging for sentences_testing set

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

### KG vector needed initializer
query_entity_dict = {}
entity_vec_dict = {}
kg = 'dbpedia'  # knowledge graph
vec_size = 200  # vector_size

###start retrieving KG vector as add-ons
print(f'start getting KG vector')
t0 = time.time()
#train_Ltitle = X_train.apply(lambda r: spacy_ent_to_vec(r['left_title'], kg, vec_size), axis=1).values
train_Lbrand = X_train.apply(lambda r: spacy_ent_to_vec(r['left_brand'], kg, vec_size), axis=1).values
t1 = time.time()
print(f'Train_Left_brand done, time cost: {t1-t0}')
#train_Rtitle = X_train.apply(lambda r: spacy_ent_to_vec(r['right_title'], kg, vec_size), axis=1).values
train_Rbrand = X_train.apply(lambda r: spacy_ent_to_vec(r['right_brand'], kg, vec_size), axis=1).values
t2 = time.time()
print(f'Train_Right_brand done, time cost: {t2-t1}')
#valid_Ltitle = X_valid.apply(lambda r: spacy_ent_to_vec(r['left_title'], kg, vec_size), axis=1).values
valid_Lbrand = X_valid.apply(lambda r: spacy_ent_to_vec(r['left_brand'], kg, vec_size), axis=1).values
t3 = time.time()
print(f'Valid_Left_brand done, time cost: {t3-t2}')
#valid_Rtitle = X_valid.apply(lambda r: spacy_ent_to_vec(r['right_title'], kg, vec_size), axis=1).values
valid_Rbrand = X_valid.apply(lambda r: spacy_ent_to_vec(r['right_brand'], kg, vec_size), axis=1).values
t4 = time.time()
print(f'Valid_Right_brand done, time cost: {t4-t3}')
#test_Ltitle = X_test.apply(lambda r: spacy_ent_to_vec(r['left_title'], kg, vec_size), axis=1).values
test_Lbrand = X_test.apply(lambda r: spacy_ent_to_vec(r['left_brand'], kg, vec_size), axis=1).values
t5 = time.time()
print(f'Test_Left_brand done, time cost: {t5-t4}')
#test_Rtitle = X_test.apply(lambda r: spacy_ent_to_vec(r['right_title'], kg, vec_size), axis=1).values
test_Rbrand = X_test.apply(lambda r: spacy_ent_to_vec(r['right_brand'], kg, vec_size), axis=1).values
t6 = time.time()
print(f'Test_Right_brand done, time cost: {t6-t5}')

### Post process after retreiving the KG vectors to the same format as word2vec vectors
train_left_kgvec = np.vstack(train_Lbrand)
train_right_kgvec = np.vstack(train_Rbrand)
valid_left_kgvec = np.vstack(valid_Lbrand)
valid_right_kgvec = np.vstack(valid_Rbrand)
test_left_kgvec = np.vstack(test_Lbrand)
test_right_kgvec = np.vstack(test_Rbrand)

### can do the experiment using KG vector only classifier
#train_kgonly = np.concatenate((train_left_kgvec, train_right_kgvec), axis=1)
#valid_kgonly = np.concatenate((valid_left_kgvec, valid_right_kgvec), axis=1)
#test_kgonly = np.concatenate((test_left_kgvec, test_right_kgvec), axis=1)

### concatenate both w2v and kg vectors
Final_train = np.concatenate((train_Lid_vec, train_Lcat_vec, train_Lbrand_vec, train_Ltitle_vec, train_Lkey_vec, train_left_kgvec,
                              train_Rid_vec, train_Rcat_vec, train_Rbrand_vec, train_Rtitle_vec, train_Rkey_vec, train_right_kgvec), axis=1)

Final_valid = np.concatenate((valid_Lid_vec, valid_Lcat_vec, valid_Lbrand_vec, valid_Ltitle_vec, valid_Lkey_vec, valid_left_kgvec,
                              valid_Rid_vec, valid_Rcat_vec, valid_Rbrand_vec, valid_Rtitle_vec, valid_Rkey_vec, valid_right_kgvec), axis=1)

Final_test = np.concatenate((test_Lid_vec, test_Lcat_vec, test_Lbrand_vec, test_Ltitle_vec, test_Lkey_vec, test_left_kgvec,
                              test_Rid_vec, test_Rcat_vec, test_Rbrand_vec, test_Rtitle_vec, test_Rkey_vec, test_right_kgvec), axis=1)

t7 = time.time()
print(f'Total time spends on pre-processing: {t7-t00}')

### start ML classifier
print(f'start training')

my_tags = ['Non-Matching', 'Matching']

from sklearn.metrics import accuracy_score, classification_report
### Logistic regression
from sklearn.linear_model import LogisticRegression
#LR = LogisticRegression(n_jobs=1, C=1e5)
t1 = time.time()
LR = LogisticRegression()
LR = LR.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = LR.predict(Final_test)
print(f'LR validation accuracy: {round(LR.score(Final_valid, Y_valid), 4)}')
print(f'LR testing accuracy: {round(LR.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred,target_names=my_tags))
print(f'LR training time: {t2-t1}')

### Random Forest
from sklearn.ensemble import RandomForestClassifier
t1 = time.time()
RF = RandomForestClassifier(n_estimators=800, min_samples_split=2, min_impurity_decrease=0.0, max_features= 'sqrt', max_depth= None, criterion='gini', bootstrap=False)
RF = RF.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = RF.predict(Final_test)
print(f'RF validation accuracy: {round(RF.score(Final_valid, Y_valid), 4)}')
print(f'RF testing accuracy: {round(RF.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred,target_names=my_tags))
print(f'RF training time: {t2-t1}')

### NN
from sklearn.neural_network import MLPClassifier
t1 = time.time()
NN = MLPClassifier(solver= 'adam', hidden_layer_sizes= (300,), early_stopping= False, activation= 'relu')
NN = NN.fit(Final_train, Y_train)
t2 = time.time()
y_test_pred = NN.predict(Final_test)
print(f'NN validation accuracy: {round(NN.score(Final_valid, Y_valid), 4)}')
print(f'NN testing accuracy: {round(NN.score(Final_test, Y_test), 4)}')
print(classification_report(Y_test, y_test_pred,target_names=my_tags))
print(f'NN training time: {t2-t1}')