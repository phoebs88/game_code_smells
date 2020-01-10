#!/usr/bin/env python
# coding: utf-8

# # k-means

# In[9]:


import os
import glob
import pandas as pd
os.chdir(r"G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\all contains git-svn-id")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
for f in all_filenames:
    print(f)
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
#combined_csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')


# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[1]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

document=[]
reader = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\combined_csv.csv')
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


# In[2]:



train_doc=document[0:10000]
len(train_doc)
len(document)


# In[8]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
train_doc=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\train_doc.csv')
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_doc)
train_doc
X


# In[5]:


true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


# In[5]:


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


# In[6]:


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind]),
    print


# In[7]:


lines_for_predicting = ["tf and idf is awesome!", "some androids is there"]
KMeans.predict(vectorizer.transform(lines_for_predicting))


# # Hierarchial clustering

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(document, method='ward'))


# # Document clustering

# In[17]:


import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import nltk
nltk.download('stopwords')


# In[18]:


stopwords = nltk.corpus.stopwords.words('english')


# In[19]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[22]:



# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

nltk.download('punkt')
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[23]:


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in document:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


# In[25]:


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# In[27]:



print (vocab_frame.head())
print
print
print
print


# In[28]:



from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

get_ipython().run_line_magic('time', 'tfidf_matrix = tfidf_vectorizer.fit_transform(document) #fit the vectorizer to synopses')

print(tfidf_matrix.shape)


# In[29]:


terms = tfidf_vectorizer.get_feature_names()


# In[30]:


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print
print


# # kmeans

# In[31]:


from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

get_ipython().run_line_magic('time', 'km.fit(tfidf_matrix)')

clusters = km.labels_.tolist()


# In[32]:



from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

#km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()


# In[33]:


films = { 'synopsis': document, 'cluster': clusters }

frame = pd.DataFrame(films, index = [clusters] , columns = ['cluster'])


# In[34]:


frame['cluster'].value_counts()


# In[35]:


grouped = frame['cluster'] #groupby cluster for aggregation purposes

grouped.mean()


# In[36]:


from __future__ import print_function

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()


# In[1]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

document=[]
reader = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\combined_csv.csv')
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


# In[2]:


len(document)


# In[ ]:




