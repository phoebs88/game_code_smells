#!/usr/bin/env python
# coding: utf-8

# stats

# In[4]:


from nltk.corpus import wordnet
syns = wordnet.synsets("performance")
print(syns)


# In[9]:


import pandas as pd
df1=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\now_no_git_svn_id.csv')
df2=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_broserquest.csv')
df3=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_command.csv')
#df4=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_openshades.csv')
df4=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_hexon.csv')
df5=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_hb_satanas.csv')
#df7=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_unvanished.csv')
#df8=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_cleverraven.csv')
#df9=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_keeperrl.csv')
#df10=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\results_stuntrally.csv')
dd=df1.groupby(['user','repo']).count()
dd.shape


# In[6]:


import pandas as pd
df1=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\thematically_tagged.csv')
df1


# In[10]:


dd=df2.groupby(['user','repo']).count()
dd


# In[11]:


dd=df3.groupby(['user','repo']).count()
dd


# In[12]:


dd=df4.groupby(['user','repo']).count()
dd


# In[13]:


dd=df5.groupby(['user','repo']).count()
dd


# In[7]:


import os
import glob
import pandas as pd
os.chdir(r"G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\dataset_100")


# In[8]:


extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


# In[9]:


#combine all files in the list
for f in all_filenames:
    print(f)
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
#combined_csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')


# In[12]:


df8=pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\short comm\dataset_100\combined_csv.csv')
dd=df8.groupby(['user','repo']).count()
df8.shape


# In[ ]:





# after observing the stats of all the repos, we will try to cluster this text corpus using lda , kmeans and other clustering technique, will try to validate by manual study. 

# # LDA

# In[6]:


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


# In[6]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

wo_commits=[]
reader = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\combined_csv.csv')
 #x = pd.DataFrame({'': [1, 2, 3], 'y': [3, 4, 5]})
df=pd.DataFrame(reader)
#dd=df['contribution_type']=='issue'
#dd
#dd=df[df['contribution_type':'issue']]
#if contribution_type is commit_message, dont include 'title', just include 'text' column as it is superset of 'match' column
#if issues, then 
#dd=pd.DataFrame(df.loc[df['contribution_type'] == 'commit_message'])
##
#for i in range(0,m):
 #   document.append(dd.iloc[i]['text'])
    
di=pd.DataFrame(df.loc[df['contribution_type'] == 'Issue'])
m,n=di.shape
for i in range(0,m):
    wo_commits.append(di.iloc[i]['content'])
    
dp=pd.DataFrame(df.loc[df['contribution_type'] == 'PullRequest'])
m,n=dp.shape
for i in range(0,m):
    wo_commits.append(dp.iloc[i]['content'])


# In[7]:



# doc2, doc3, doc4, doc5]

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
train_doc=document
f_lda(train_doc)
#train_doc
#https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb


# now we need to decide upon the topics we would want to categorize our corpus, so do manual study of as much 

# In[9]:



# doc2, doc3, doc4, doc5]

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
train_doc=wo_commits
f_lda(train_doc)


# In[4]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
f_lda(train_doc)


# In[5]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
f_lda(train_doc)


# In[6]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
f_lda(train_doc)


# In[7]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    ldamodel = Lda(doc_term_matrix, num_topics=7, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
f_lda(train_doc)


# In[ ]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
#import gensim
#nltk.download('stopwords')
#nltk.download('wordnet')


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
    ldamodel = Lda(doc_term_matrix, num_topics=9, id2word = dictionary, passes=50)
    
    new_str = ldamodel.print_topics(num_topics = 5, num_words=30)
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
        print("topic:")
        print(topk)
        t_word = [w for w,_ in topk]
        print("t_word:")
        print(t_word)
        
    return t_word
f_lda(train_doc)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print("-----------------")
            print (documents[doc_index])

#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = document

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 5

# Run NMF
nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_W = nmf_model.transform(tfidf)
nmf_H = nmf_model.components_

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

no_top_words = 10
no_top_documents = 5
display_topics(nmf_H, nmf_W, tfidf_feature_names, documents, no_top_words, no_top_documents)
display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print("-----------------")
            print (documents[doc_index])

#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = wo_commits

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 5

# Run NMF
nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_W = nmf_model.transform(tfidf)
nmf_H = nmf_model.components_

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

no_top_words = 10
no_top_documents = 5
display_topics(nmf_H, nmf_W, tfidf_feature_names, documents, no_top_words, no_top_documents)
display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)


# In[32]:


tfidf


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print("-------------")
            print (documents[doc_index])

#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = train_doc

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 7

# Run NMF
nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_W = nmf_model.transform(tfidf)
nmf_H = nmf_model.components_

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

no_top_words = 10
no_top_documents = 5
display_topics(nmf_H, nmf_W, tfidf_feature_names, documents, no_top_words, no_top_documents)
display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)


# # above was using tf-idf, now using bag of words

# In[ ]:





# In[25]:


'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


# In[32]:


import gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import pandas as pd
stemmer = SnowballStemmer("english")
#nltk.download('wordnet')
processed_docs = []

for doc in train_doc:
    processed_docs.append(preprocess(doc))
processed_docs


# # bag of words

# In[33]:


dictionary = gensim.corpora.Dictionary(processed_docs)


# In[34]:


count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[35]:


'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)


# In[36]:


'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[37]:


'''
Preview BOW for our sample preprocessed document
'''
document_num = 20
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))


# In[38]:


lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)


# In[39]:


for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


# # testing on unseen doc

# In[41]:


unseen_document='i have iisue in synchronising the game play during the server multiplayer game.'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# # semantic clustering of text corpus

# In[3]:


import pytorch


# In[4]:


conda install -c pytorch pytorch


# In[1]:


from random import randint

import numpy as np
import torch
import torch


# In[2]:


import nltk
import sklearn

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[3]:


import nltk
#nltk.download('punkt')


# In[4]:


pwd


# In[10]:


#from . import models
from models import InferSent
V = 1
MODEL_PATH = 'Documents/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))


# In[19]:


W2V_PATH = 'Documents/FastText/crawl-300d-2M.vec/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)


# In[20]:


infersent.build_vocab(train_doc, tokenize=True)


# In[21]:


embeddings = infersent.encode(train_doc, tokenize=True)


# In[22]:


infersent.visualize('A man plays an instrument.', tokenize=True)


# In[31]:


embeddings.shape
#This outputs a numpy array with n vectors of dimension 4096.


# Now we have each and every sentence in the form of vector. We can try to match up each to some of our topics or we can try to apply clustering to see the distribution.

# # k nn clustering applied 

# In[34]:


from sklearn.cluster import KMeans
import numpy as np
#X = np.array([[1, 2], [1, 4], [1, 0],
 #             [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
kmeans.labels_

#kmeans.predict([[0, 0], [12, 3]])

kmeans.cluster_centers_


# In[39]:


import matplotlib.pyplot as plt
plt.scatter(
    embeddings[y_km == 0, 0], embeddings[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    embeddings[y_km == 1, 0], embeddings[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

'''plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3')'''

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# In[29]:


import nmslib

NTHREADS = 8
def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index
def get_knns(index, vecs, k):
    return zip(*index.knnQueryBatch(vecs, k=k,num_threads=NTHREADS))

nn_wvs = create_index(embeddings)

to_frame = lambda x: pd.DataFrame(np.array(x)[:,1:])

idxs, dists = map(to_frame, get_knns(nn_wvs, embeddings, 10))

catted = pd.concat([idxs.stack().to_frame('idx'), 
                    dists.stack().to_frame('dist')], 
                   axis=1).reset_index().drop('level_1',1).rename(columns={'level_0': 'v1', 'idx': 'v2'})


# In[32]:


nn_wvs
#colormap = np.array(['Red', 'Blue', 'Green'])
#z = plt.scatter(x.sepal_length, x.sepal_width, x.petal_length, c = colormap[model.labels_])


# In[24]:


Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


model = KMeans(n_clusters = 3)
model.fit(x)
model.labels_
colormap = np.array(['Red', 'Blue', 'Green'])
z = plt.scatter(x.sepal_length, x.sepal_width, x.petal_length, c = colormap[model.labels_])

