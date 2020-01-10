#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


# In[8]:


df = pd.read_csv("lda_model6_ques.csv")

# Good distribution 
# models 2, 6

df


# In[22]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['Body'].fillna(''))
#x = v.fit_transform(df['Review'].values.astype('U'))
#features = tfidf.fit_transform(df['Body'].apply(lambda x: np.str_(x)))
labels = df.Topics


# In[23]:


# cross_val_score(LinearSVC(), features, labels, scoring='accuracy', cv=2)
# TFIDF array([0.7877289 , 0.78578947])
# CountVectorizer array([0.77299516, 0.77189474])


# In[28]:



X_train, X_test, y_train, y_test = train_test_split(df['Body'].fillna(''), df['Topics'], random_state = 0, test_size=0.20)

count_vect = CountVectorizer()

count_vect.fit(df['Body'].fillna(''))
X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)


clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)
y_predicted = clf.predict(X_test_tfidf)


# In[30]:


# print(clf.predict(tfidf.transform(["Error"])))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_predicted,y_test)*100)


# In[52]:


import pickle
with open("ml_model.mdl", "wb") as wf:
    pickle.dump(clf, wf)

