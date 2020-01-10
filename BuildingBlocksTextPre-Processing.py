#!/usr/bin/env python
# coding: utf-8

# ## Building Blocks: Text Pre-Processing
# 
# This article is the second of more to come articles on Natural Language Processing. The purpose of this series of articles is to document my journey as I learn about this subject, as well as help others gain efficiency from it.
# 
# In the last article of our series, we introduced the concept of Natural Language Processing, you can read it here, and now you probably want to try it yourself, right? Great! Without further ado, let's dive in to the building blocks for statistical natural language processing. 
# 
# In this article, we'll introduce the key concepts, along with practical implementation in Python and the challenges to keep in mind at the time of application.
# ** **

# ### Text Normalization
# 
# Normalizing the text means converting it to a more convenient, standard form before performing turning it to features for higher level modeling. Think of this step as converting human readable language into a form that is machine readable.
# 
# The standard framework to normalize the text includes:
# 1. Tokenization
# 2. Stop Words Removal
# 3. Morphological Normalization
# 4. Collocation
# 
# Data preprocessing consists of a number of steps, any number of which may or not apply to a given task. More generally, in this article we'll discuss some predetermined body of text, and perform some basic transformative analysis that can be used for performing further, more meaningful natural language processing
# 
# ** **
# #### Tokenization
# 
# Given a character sequence and a defined document unit (blurb of texts), tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters/words, such as punctuation. Ordinarily, there are two types of tokenization:
# 
# 1. Word Tokenization: Used to separate words via unique space character. Depending on the application, word tokenization may also tokenize multi-word expressions like New York. This is often times is closely tied to a process called Named Entity Recognition. Later in this tutorial, we will look at Collocation (Phrase) Modeling that helps address part of this challenge
# 
# 2. Sentence Tokenization/Segmentation: Along with word tokenization, sentence segmentation is a crucial step in text processing. This is usually performed based on punctuations such as ".", "?", "!" as they tend to mark the sentence boundaries
# 
# **Challenges:**
# - The use of abbreviations may prompt the tokenizer to detect a sentence boundary where there is none. 
# - Numbers, special characters, hyphenation, and capitalization. In the expressions "don't," "I'd," "John's" do we have one, two or three tokens?

# In[1]:


from nltk.tokenize import sent_tokenize, word_tokenize

#Sentence Tokenization
print ('Following is the list of sentences tokenized from the sample review\n')

sample_text = """The first time I ate here I honestly was not that impressed. I decided to wait a bit and give it another chance. 
I have recently eaten there a couple of times and although I am not convinced that the pricing is particularly on point the two mushroom and 
swiss burgers I had were honestly very good. The shakes were also tasty. Although Mad Mikes is still my favorite burger around, 
you can do a heck of a lot worse than Smashburger if you get a craving"""

tokenize_sentence = sent_tokenize(sample_text)

print (tokenize_sentence)
print ('---------------------------------------------------------\n')
print ('Following is the list of words tokenized from the sample review sentence\n')
tokenize_words = word_tokenize(tokenize_sentence[1])
print (tokenize_words)


# ** **
# #### Stop Words Removal
# Often, there are a few ubiquitous words which would appear to be of little value in helping the purpose of analysis but increases the dimensionality of feature set, are excluded from the vocabulary entirely as the part of stop words removal process. There are two considerations usually that motivate this removal.
# 
# 1. Irrelevance: Allows one to analyze only on content-bearing words. Stopwords, also called empty words because they generally do not bear much meaning, introduce noise in the analysis/modeling process
# 2. Dimension: Removing the stopwords also allows one to reduce the tokens in documents significantly, and thereby decreasing feature dimension
# 
# **Challenges:**
# 
# Converting all characters into lowercase letters before stopwords removal process can introduce ambiguity in the text, and sometimes entirely changing the meaning of it. For example, with the expressions "US citizen" will be viewed as "us citizen" or "IT scientist" as "it scientist". Since both *us* and *it* are normally considered stop words, it would result in an inaccurate outcome. The strategy regarding the treatment of stopwords can thus be refined by identifying that "US" and "IT" are not pronouns in the above examples, through a part-of-speech tagging step.

# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# define the language for stopwords removal
stopwords = set(stopwords.words("english"))
print ("""{0} stop words""".format(len(stopwords)))

tokenize_words = word_tokenize(sample_text)
filtered_sample_text = [w for w in tokenize_words if not w in stopwords]

print ('\nOriginal Text:')
print ('------------------\n')
print (sample_text)
print ('\n Filtered Text:')
print ('------------------\n')
print (' '.join(str(token) for token in filtered_sample_text))


# ** **
# #### Morphological Normalization
# Morphology, in general, is the study of the way words are built up from smaller meaning-bearing units, morphomes. For example, dogs consists of two morphemes: dog and s
# 
# Two commonly used techniques for text normalization are:
# 
# 1. Stemming: The procedure aims to identify the stem of a word and use it in lieu of the word itself. The most popular algorithm for stemming English, and one that has repeatedly been shown to be empirically very effective, is Porter's algorithm. The entire algorithm is too long and intricate to present here, but you can find details here
# 2. Lemmatization: This process refers to doing things correctly with the use of vocabulary and morphological analysis of words, typically aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.
# 
# If confronted with the token saw, stemming might return just s, whereas lemmatization would attempt to return either see or saw depending on whether the use of the token was as a verb or a noun

# In[3]:


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

tokenize_words = word_tokenize(sample_text)

stemmed_sample_text = []
for token in tokenize_words:
    stemmed_sample_text.append(ps.stem(token))

lemma_sample_text = []
for token in tokenize_words:
    lemma_sample_text.append(lemmatizer.lemmatize(token))
    
print ('\nOriginal Text:')
print ('------------------\n')
print (sample_text)

print ('\nFiltered Text: Stemming')
print ('------------------\n')
print (' '.join(str(token) for token in stemmed_sample_text))

print ('\nFiltered Text: Lemmatization')
print ('--------------------------------\n')
print (' '.join(str(token) for token in lemma_sample_text))


# ** **
# **Challenges:**
# 
# Often, full morphological analysis produces at most very modest benefits for analysis. Neither form of normalization improve language information performance in aggregate, both from relevance and dimensionality reduction standpoint - at least not for the following situations:

# In[4]:


from nltk.stem import PorterStemmer
words = ["operate", "operating", "operates", "operation", "operative", "operatives", "operational"]

ps = PorterStemmer()

for token in words:
    print (ps.stem(token))


# ** **
# As an example of what can go wrong, note that the Porter stemmer stems all of the following words to oper
# However, since operate in its various forms is a common verb, we would expect to lose considerable precision:
# - operational AND research
# - operating AND system
# - operative AND dentistry
# 
# For cases like these, moving to using a lemmatizer would not completely fix the problem because particular inflectional forms are used in specific collocations. Getting better value from term normalization depends more on pragmatic issues of word use than on formal issues of linguistic morphology
