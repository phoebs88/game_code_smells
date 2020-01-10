#!/usr/bin/env python
# coding: utf-8

# In[104]:


# Run LDA over all the documents to get required number of topics
# Then tag every document using topic-correlation matrix
# Create a Machine learning model over the entire vectorized post with an added target column
# Given a new document, vectorize the document and predict label using ML model


# In[21]:


import pandas as pd
import numpy as np
import gensim
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

import spacy
from pprint import pprint


# In[22]:


# First phase is to create topic-correlation matrix using LDA

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
exceptions = ['what','why','how','when','where','to','can','could','should']
stop_words = list(set(stop_words) - set(exceptions))
stop_words.extend(['www','com','org','color','html','system','aspx','png','alt','image','strong','nofollow','rel','http','xa','noreferrer','href','p','pre','li','ol','nbsp','fldpi','time','en_wikipedia','sql','table','child','sqlconnection','sqlstatement'])


# In[38]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

document=[]
reader = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\dataset_100\combined_csv.csv')
 #x = pd.DataFrame({'': [1, 2, 3], 'y': [3, 4, 5]})
df=pd.DataFrame(reader)
#dd=df['contribution_type']=='issue'
#dd
#dd=df[df['contribution_type':'issue']]
#if contribution_type is commit_message, dont include 'title', just include 'text' column as it is superset of 'match' column
#if issues, then 
dd=pd.DataFrame(df.loc[df['contribution_type'] == 'commit_message'])
m,n=dd.shape
for i in range(0,m):
    document.append(dd.iloc[i]['text'])
    
di=pd.DataFrame(df.loc[df['contribution_type'] == 'Issue'])
m,n=di.shape
for i in range(0,m):
    document.append(di.iloc[i]['content'])
    
dp=pd.DataFrame(df.loc[df['contribution_type'] == 'PullRequest'])
m,n=dp.shape
for i in range(0,m):
    document.append(dp.iloc[i]['content'])
#for i in range(0,m):
    #df.loc[df['contribution_type'] == 'Issue']:
#document.append(dd['title']+" "+dd['text'])


#li=list(document)

#type(document)
#document.to_csv("pr.csv")

#df.filter(["title", "text","contribution_type"]) 

df = pd.DataFrame(document, columns = ['Body']) 
  
#df=pd.DataFrame(document)
df


# In[39]:


data = df.Body.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))


# In[40]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[41]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[42]:


data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
nlp =spacy.load('en_core_web_sm')
#nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[43]:


id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1])


# In[44]:


# Tweak these values to get new LDA Model

# lda_model = LdaModel(corpus=corpus,
#                        id2word=id2word,
#                        num_topics=6,  
#                        chunksize=2000,
#                        passes=50,
#                        alpha='auto',
#                        per_word_topics=True)


# In[45]:


# lda_model2 = LdaModel(corpus=corpus,
#                        id2word=id2word,
#                        num_topics=6,                                           
#                        random_state=100,
#                        update_every=1,
#                        chunksize=1000,
#                        passes=50,
#                        alpha='auto',
#                        per_word_topics=True)


# In[46]:


lda_model3 = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=8,                                           
                       random_state=50,
                       update_every=1,
                       chunksize=1500,
                       passes=20,
                       alpha='auto',
                       per_word_topics=True)


# In[47]:


pprint(lda_model3.print_topics(num_words = 15))
doc_lda = lda_model3[corpus]


# In[48]:


document_topics = []
for topic in doc_lda:
    k = sorted(topic[0], reverse=True, key=lambda k: k[1])[0]
    document_topics.append(k)


# In[49]:


np_doc_topics = np.array(document_topics)
np_doc_topics


# In[50]:


df['Topics'] = np_doc_topics[:,0]
df['Probabilities'] = np_doc_topics[:,1]


# In[51]:


df.Topics.value_counts()
df


# In[52]:


df.to_csv("lda_model6_ques.csv")


# In[53]:


import pickle
from time import time
with open("MODEL/lda_model_6_ques" + ".mdl", "wb") as wf:
    pickle.dump(lda_model3, wf)


# In[393]:


# from gensim.test.utils import datapath
# wf = datapath("MODEL/native_model_" + "6_ques")
# lda_model3.save(wf)

# lda = LdaModel.load(wf)


# In[190]:


# from gensim.corpora.dictionary import Dictionary
# from gensim.test.utils import common_texts
# common_dictionary = Dictionary(common_texts)
# common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# other_texts = [
#      ['why','problem'],
#     ['system','issue']
#  ]
# other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]

# unseen_doc = other_corpus[0]
# print(unseen_doc)
# vector = lda_model[unseen_doc]


# In[ ]:




