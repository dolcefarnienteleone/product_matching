import warnings
import time
import re

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, \
    remove_special_character, check_spelling, remove_stopword, tokenize_word, stem_word
import nltk

preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, remove_special_character,
                        check_spelling, remove_stopword, lemmatize_word]
def clean_text(text):
    # Cleaing the text
    text = re.sub('@en', '', text)
    text = re.sub('@es', '', text)
    text = re.sub('@fr', '', text)
    text = preprocess_text(text, preprocess_functions)

    return text


corpus = open('train_corpus_v4.txt', 'r')
raw = corpus.read()

raw = clean_text(raw)
t1 = time.time()
# iterate through each sentence in the file
all_sentences = nltk.sent_tokenize(raw)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

t2 = time.time()
#t2 = time.time()
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

### save model
model_CBOW.save('w2v_cbow_v4.model')
model_SG.save('w2v_sg_v4.model')