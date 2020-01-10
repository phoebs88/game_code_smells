#!/usr/bin/env python
# coding: utf-8

# ### Evaluate Topic Model in Python: Latent Dirichlet Allocation (LDA)
# 
# In the previous article, I introduced the concept of topic modeling and walked through the code for developing your first topic model using Latent Dirichlet Allocation (LDA) method in the python using sklearn implementation.
# 
# Pursuing on that understanding, in this article, I'll go a few steps deeper by outlining the framework to quantitatively evaluate topic models through the measure of topic coherence and share the code template in python using Gensim implementation to allow for end-to-end model development.
# 
# ### Why evaluate topic models?
# 
# ![img](https://tinyurl.com/y3xznjwq)
# 
# We know probabilistic topic models, such as LDA, are popular tools for analysis of the text, providing both a predictive and latent topic representation of the corpus. There is a longstanding assumption that the latent space discovered by these models is meaningful and useful, and evaluating such assumptions is challenging due to its unsupervised training process. There is a no-gold standard list of topics to compare against every corpus.
# 
# However, it is equally important to identify if a trained model is objectively good or bad, as well have an ability to compare different models/methods and to do so, we require an objective measure for the quality. Traditionally, and still for many practical applications, to evaluate if "the correct thing" has been learned about the corpus, we use implicit knowledge and "eyeballing" approaches. Ideally, we'd like to capture this information in a single metric that can be maximized, and compared. Let's take a look at roughly what approaches are commonly used for the evaluation:
# 
# **Eye Balling Models**
# - Top N words
# - Topics / Documents
# 
# **Intrinsic Evaluation Metrics**
# - Capturing model semantics
# - Topics interpretability
# 
# **Human Judgements**
# - What is a topic
# 
# **Extrinsic Evaluation Metrics/Evaluation at task**
# - Is model good at performing predefined tasks, such as classification
# 
# Natural language is messy, ambiguous and full of subjective interpretation, and sometimes trying to cleanse ambiguity reduces the language to an unnatural form. Nevertheless, in this article, we'll explore more about topic coherence, and how we can use it to quantitatively justify the model selection.
# 
# ### What is Topic Coherence?
# 
# Perplexity is often used as an example of an intrinsic evaluation measure. It comes from the language modeling community and aims to capture how surprised a model is of new data it has not seen before. It is measured as the normalized log-likelihood of a held-out test set.
# 
# Focussing on the log-likelihood part, you can think of the perplexity metric as measuring how probable some new unseen data is given the model that was learned earlier. That is to say, how well does the model represent or reproduce the statistics of the held-out data.
# 
# However, past research has shown that predictive likelihood (or equivalently, perplexity) and human judgment are often not correlated, and even sometimes slightly anti-correlated. And that served as a motivation for more work trying to model the human judgment, and thus `Topic Coherence`.
# 
# The topic coherence concept combines a number of papers into one framework that allows evaluating the coherence of topics inferred by a topic model. But,
# 
# #### What is topic coherence?
# Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and topics that are artifacts of statistical inference. But,
# 
# #### What is coherence?
# A set of statements or facts is said to be coherent, if they support each other. Thus, a coherent fact set can be interpreted in a context that covers all or most of the facts. An example of a coherent fact set is "the game is a team sport", "the game is played with a ball", "the game demands great physical efforts"
# 
# ### Coherence Measures
# 
# 1. `C_v` measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
# 2. `C_p` is based on a sliding window, one-preceding segmentation of the top words and the confirmation measure of Fitelson's coherence
# 3. `C_uci` measure is based on a sliding window and the pointwise mutual information (PMI) of all word pairs of the given top words
# 4. `C_umass` is based on document cooccurrence counts, a one-preceding segmentation and a logarithmic conditional probability as confirmation measure
# 5. `C_npmi` is an enhanced version of the C_uci coherence using the normalized pointwise mutual information (NPMI)
# 6. `C_a` is baseed on a context window, a pairwise comparison of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity

# #### Install dependent packages

# In[4]:


# To be run only once
#if 0 == 1:
get_ipython().system('pip install gensim')
get_ipython().system('pip install PyLDAvis')
get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# ### Model Implementation
# 
# #### Loading data
# 
# For this tutorial, we’ll use the dataset of papers published in NIPS conference. The NIPS conference (Neural Information Processing Systems) is one of the most prestigious yearly events in the machine learning community. The CSV data file contains information on the different NIPS papers that were published from 1987 until 2016 (29 years!). These papers discuss a wide variety of topics in machine learning, from neural networks to optimization methods, and many more.
# 
# <img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_158/img/nips_logo.png" alt="The logo of NIPS (Neural Information Processing Systems)">
# 
# Let’s start by looking at the content of the file

# In[5]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

document=[]
reader = pd.read_csv(r'/storage/chimalakonda/game_code_smells/results_adarkroom.csv')
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
document.to_csv("pr.csv")

#df.filter(["title", "text","contribution_type"]) 


# In[15]:


df = pd.DataFrame(document, columns=['Text']) 


# In[16]:


# Importing modules
import pandas as pd
import os

os.chdir('..')

# Read data into papers
#papers = pd.read_csv(r'G:\software_engg_projects\code_smells_in_games\Code smells journal\scrapr_regex_result\no error\combined_csv.csv')

# Print head
df.head()


# #### Data Cleaning
# 
# Since the goal of this analysis is to perform topic modeling, we will solely focus on the text data from each paper, and drop other metadata columns

# In[43]:


# Remove the columns
#papers = papers.drop(columns=['user', 'title', 'abstract', 
                              #'event_type', 'pdf_name', 'year'], axis=1)

# sample only 100 papers
#papers = df.sample(100)
papers= df

# Print out the first rows of papers
papers.head()


# #### Remove punctuation/lower casing
# 
# Next, let’s perform a simple preprocessing on the content of paper_text column to make them more amenable for analysis, and reliable results. To do that, we’ll use a regular expression to remove any punctuation, and then lowercase the text

# In[44]:


# Load the regular expression library
import re

# Remove punctuation
papers['paper_text_processed'] = papers['Text'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

# Print out the first rows of papers
papers['paper_text_processed'].head()


# #### Tokenize words and further clean-up text
# 
# Let’s tokenize each sentence into a list of words, removing punctuations and unnecessary characters altogether.

# In[45]:


import gensim
from gensim.utils import simple_preprocess

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))

print(data_words[:1][0][:30])


# #### Creating Bigram and Trigram Models
# 
# Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring. Some examples in our example are: 'back_bumper', 'oil_leakage', 'maryland_college_park' etc.
# 
# Gensim's Phrases model can build and implement the bigrams, trigrams, quadgrams and more. The two important arguments to Phrases are min_count and threshold.

# In[46]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# #### Remove Stopwords, Make Bigrams and Lemmatize
# 
# The phrase models are ready. Let’s define the functions to remove the stopwords, make trigrams and lemmatization and call them sequentially.

# In[47]:


# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[48]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
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


# Let's call the functions in order.

# In[49]:


import spacy

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1][0][:30])


# ### Data transformation: Corpus and Dictionary
# 
# The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus. Let’s create them.

# In[50]:


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])


# ### Building the base topic model
# 
# We have everything required to train the base LDA model. In addition to the corpus and dictionary, you need to provide the number of topics as well. Apart from that, alpha and eta are hyperparameters that affect sparsity of the topics. According to the Gensim docs, both defaults to 1.0/num_topics prior (we'll use default for the base model).
# 
# chunksize controls how many documents are processed at a time in the training algorithm. Increasing chunksize will speed up training, at least as long as the chunk of documents easily fit into memory.
# 
# passes controls how often we train the model on the entire corpus (set to 10). Another word for passes might be "epochs". iterations is somewhat technical, but essentially it controls how often we repeat a particular loop over each document. It is important to set the number of "passes" and "iterations" high enough.

# In[51]:


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)


# #### View the topics in LDA model
# The above LDA model is built with 10 different topics where each topic is a combination of keywords and each keyword contributes a certain weightage to the topic.
# 
# You can see the keywords for each topic and the weightage(importance) of each keyword using `lda_model.print_topics()`

# In[52]:


from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# #### Compute Model Perplexity and Coherence Score
# 
# Let's calculate the baseline coherence score

# In[53]:


from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# ### Hyperparameter tuning
# 
# First, let's differentiate between model hyperparameters and model parameters :
# 
# - `Model hyperparameters` can be thought of as settings for a machine learning algorithm that are tuned by the data scientist before training. Examples would be the number of trees in the random forest, or in our case, number of topics K
# 
# - `Model parameters` can be thought of as what the model learns during training, such as the weights for each word in a given topic.
# 
# Now that we have the baseline coherence score for the default LDA model, let's perform a series of sensitivity tests to help determine the following model hyperparameters: 
# - Number of Topics (K)
# - Dirichlet hyperparameter alpha: Document-Topic Density
# - Dirichlet hyperparameter beta: Word-Topic Density
# 
# We'll perform these tests in sequence, one parameter at a time by keeping others constant and run them over the two difference validation corpus sets. We'll use `C_v` as our choice of metric for performance comparison 

# In[54]:


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


# Let's call the function, and iterate it over the range of topics, alpha, and beta parameter values

# In[1]:


import numpy as np
import tqdm

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
               corpus]

corpus_title = ['100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()


# In[39]:


model=pd.DataFrame(model_results)
model.to_csv("lda_tuning_results.csv")


# ### Final Model Training
# 
# Based on external evaluation (Code to be added from Excel based analysis), train the final model

# In[40]:


lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.9)


# In[41]:


from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[42]:


import pyLDAvis.gensim
import pickle 
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

LDAvis_prepared


# In[ ]:




