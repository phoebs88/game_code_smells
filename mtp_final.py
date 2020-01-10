#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df1=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\thematically_tagged.csv')
df1


# In[2]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

document=[]
reader = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\thematically_tagged.csv')
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


# In[3]:


document


# In[12]:


import nltk
import gensim
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
import pandas as pd

def f_lda(doc_complete):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]
    # Importing Gensim
    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    import gensim
    from gensim import corpora
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=8, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = -1, num_words=30)
    print(new_str)

    # print("NUMBER OF TOPICS : ")
    # print(ldamodel.num_topics)0
    # for t in range(ldamodel.num_topics):
    #     topk = ldamodel.show_topic(t,29)
    #     print("list of topics -----------------")
    #     print(topk)


    var_word = ""

    for t in range(ldamodel.num_topics):
        topk = ldamodel.show_topic(t,5)
        print("topic:", t)
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
train_doc=document
f_lda(train_doc)


# In[15]:


from pprint import pprint
pprint(ldamodel.print_topics(num_words = 15))
doc_lda = ldamodel[corpus]


# In[ ]:


document_topics = []
for topic in doc_lda:
    k = sorted(topic[0], reverse=True, key=lambda k: k[1])[0]
    document_topics.append(k)


# In[ ]:


np_doc_topics = np.array(document_topics)


# In[ ]:


df['Topics'] = np_doc_topics[:,0]
df['Probabilities'] = np_doc_topics[:,1]


# In[ ]:


df.Topics.value_counts()


# In[ ]:




