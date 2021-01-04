import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import string
import time
import re

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

corpus = open('train_corpus_v4.txt', 'r')
raw = corpus.read()

def clean_text(text):
    # Cleaing the text
    #text = re.sub('\[.*? @\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)

    return text

#r = clean_text(raw)
t1 = time.time()
# iterate through each sentence in the file
all_sentences = nltk.sent_tokenize(raw)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

t2 = time.time()
# Create CBOW model
model_CBOW = gensim.models.Word2Vec(all_words, min_count=1,
                                    size=25, window=5, iter=3)
t3 = time.time()
# Create Skip Gram model
model_SG = gensim.models.Word2Vec(all_words, min_count=1,
                                  size=25, window=5, iter=3, sg=1)
t4 = time.time()

print(f'tokenize spend: {t2-t1}')
print(f'Training CBOW model spend: {t3-t2}')
print(f'Training SG model spend: {t4-t3}')

#save CBOW model
#filename_CBOW = 'rawtrain_embed_w2v_cbow.txt'
#model_CBOW.wv.save_word2vec_format(filename_CBOW, binary=False)
model_CBOW.save('w2v_cbow_v3.model')
model_SG.save('w2v_sg_v3.model')