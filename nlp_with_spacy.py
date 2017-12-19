#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:18:51 2017

@author: stephanosarampatzes
"""

# import libraries

from bs4 import BeautifulSoup
import requests
import urllib.request

import re
import pandas as pd
import numpy as np

# extract urls from web pages
def getLinks(url, magicFrag, token):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    numErrors = 0
    
    for link in soup.find_all('a', attrs = {'href': re.compile("^https://")}):
        try:
            url = link['href']
            if ((magicFrag is not None and magicFrag in url) or magicFrag is None):
                links.append(link.get('href'))
        except:
            numErrors += 1
    return(links)


# the sources for urls we need
source = ["https://www.washingtonpost.com/sports", "https://www.washingtonpost.com/business/technology",
          "http://www.nytimes.com/pages/technology/index.html", "https://www.nytimes.com/pages/sports/index.html"]

# an empty list to store urls
links = []
for src in source:
    if re.search(r'(/sports)', source[0]):
        getLinks(src, '2017', 'sports')
    else:    
        getLinks(src, '2017', 'technology')
        
# we may have duplicates. We don't want this to happen
unique_links = list(set(links))

# first Sunday
stored_links = pd.DataFrame(unique_links,columns=['url'])
stored_links.to_csv('url_links.csv', encoding='utf-8')

# second Sunday
stored_links2 = pd.DataFrame(unique_links,columns=['url'])
stored_links2.to_csv('url_links2.csv', encoding='utf-8')

df = pd.read_csv('url_links.csv').append(pd.read_csv('url_links2.csv'), ignore_index = True)

# be sure once again for unique urls

df = pd.DataFrame(df['url'].unique(), columns = ['url'])

# Washington Post text extraction
def getWPText(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find_all('article', attrs = {'class' : "paywall"})
    text = ''.join([n.get_text() for n in text])
    return(text, soup.title.text)
    
# New York Times text extraction
def getNYText(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find_all('p', attrs={'class':'story-body-text story-content'})
    text = ''.join([n.get_text() for n in text])
    return(text, soup.title.text)


# import some usefull librabries like nlargest
from collections import defaultdict
from string import punctuation
from heapq import nlargest
# imprort spacy and en_core_web_sm model for English
import spacy
import en_core_web_sm as en
#import ENGLISH_STOP_WORDS to remove a bunch of stop words to reduce the size of our vocabulary
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

#load spaCy model
nlp = en.load()

class Freq_Sum:
    def __init__(self, min_cut = 0.1, max_cut = 0.9):
        # Initilize the text summarizer.
        # Words that have a frequency term lower than min_cut 
        # or higer than max_cut will be ignored.
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(list(ENGLISH_STOP_WORDS) + list(punctuation)+["n't","'d"
                              "'s", "'m", "--", "---", "...", "“", "”", "'ve"])
        
    def _compute_frequencies(self, word_sent):
        # Compute the frequency of each of word
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        # frequencies normalization and fitering
        m = float(max(freq.values()))
        try:
            for w in freq.keys():
                freq[w] = freq[w]/m
                if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                    del freq[w]
        except RuntimeError:
            pass
        return freq
    
    def summarize(self, text, n):
        # Return a list of n sentences which represent the summary of text.
        text = nlp(text)
        sents = [sent for sent in text.sents]     
        assert n <= len(sents)
        # lemmatisation
        tokens = [t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_ for t in sents]
        tokens = [t for t in tokens if t not in self._stopwords]
        self._freq = self._compute_frequencies(tokens)
        ranking = defaultdict(int)
        for i , sent in enumerate(tokens):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sents_idx = self._rank(ranking, n)
        return([sents[j] for j in sents_idx])
      
    def _rank(self, ranking, n):
        return(nlargest(n, ranking, key = ranking.get))


# drop links with videos and Q&A section with mr. Boswell
df = df[~df.url.str.contains('(^https://www.washingtonpost.com/video/)')]
df = df[~df.url.str.contains('(ask-boswell)')]

# These links break assertion. 
#!CAUTION! these indices were wrong for me. For a new HTML parse and url storage you should place your wrong ones :)
df.drop(df.index[145], inplace = True)
df.drop(df.index[230], inplace = True)
df.drop(df.index[233], inplace = True)
df.drop(df.index[233], inplace = True)
df.drop(df.index[249], inplace = True)
df.drop(df.index[316], inplace = True)

# new empty Dataframe
final_df = pd.DataFrame(columns=['title','text','url'])

fs = Freq_Sum()

# keep 5 sentences from every text 
for i in df.index:
    if re.search(r'(washingtonpost)', df.url[i]):
        try:
            #print(i, df.index[i], df.url[i]) this to find broken links
            store_text = getWPText(df.url[i])
            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 5)), df.url[i]
        # pass the missing indices
        except KeyError:
            pass
    else:
        try:
            #print(i, df.index[i], df.url[i]) this to find broken links
            store_text = getNYText(df.url[i])
            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 5)), df.url[i]
        # pass the missing indices
        except KeyError:
            pass
        except AssertionError:
            store_text = getNYText(df.url[i])
            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 2)), df.url[i]

#for i in range(,):
#    if re.search(r'(washingtonpost)', df.url[i]):
#        try:
#            store_text = getWPText(df.url[i])
#            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 5)), df.url[i]
#        except KeyError:
#            pass
#    else:
#        try:
#            store_text = getNYText(df.url[i])
#            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 5)), df.url[i]
#        except KeyError:
#            pass
#        except AssertionError:
#            store_text = getNYText(df.url[i])
#            final_df.loc[i] = store_text[1], str(fs.summarize(store_text[0], 2)), df.url[i]


#re-import the final dataset 
final_df = pd.read_csv('final_texts.csv')
final_df = final_df.drop(['Unnamed: 0'], axis =1)
print('size of dataset: ', final_df.shape)
final_df.head()

# replace '\xa0' with a white space.
final_df['text'] = final_df['text'].apply(lambda s: s.replace(u'\xa0', u' '))

# create target value
final_df['news'] = ''

for j in final_df.index:
    if (re.search(r'(sport)', final_df['url'][j]) or re.search(r'(/early-lead)', final_df['url'][j]) 
    or re.search(r'(-insider)', final_df['url'][j]) or re.search(r'(fancy-stats)', final_df['url'][j])):
        final_df['news'][j] = 'sports'
    else:
        final_df['news'][j] = 'tech'

# get dummies        
final_df['dummy_news'] = pd.get_dummies(final_df.news, drop_first = True)
final_df.head()


# clean and proceed to model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# classifier
from sklearn.naive_bayes import MultinomialNB


class TextCleaner(TransformerMixin):
    
    def transform(self, X, **transform_params):
        return(cleanText(text) for text in X)
        
    def fit(self, X, y = None, **fit_params):
        return(self)

    def get_params(self, deep = True):
        return {}

def cleanText(text):
    #remove \n's and \r's
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    text = text.lower()
    return(text)
    
def text_tokenizer(sentence):
    # remove any character that is not letter
    sentence = re.sub('[^a-zA-z]', ' ', sentence)
    toks = nlp(sentence)
    toks = [t.lemma_.lower().strip() if t.lemma_ != 'PRON' else t.lower_ for t in toks]    
    toks = [t for t in toks if t not in set(list(ENGLISH_STOP_WORDS) + list(punctuation)+[
            "n't","'d","'s", "'m", "--", "---", "...", "“", "”", "'ve"])]
    # drop any words that have less than 2 characters
    toks = [t for t in toks if len(t)>2]
    return(toks)  
    
vect = CountVectorizer(tokenizer = text_tokenizer, ngram_range = (1,1))

mnb = MultinomialNB()

pipe = Pipeline([('cleaner', TextCleaner()),
                 ('vectorizer', vect),
                 ('classifier', mnb)])
# shuffle rows
final_df.sample(frac=1)

from sklearn.model_selection import train_test_split
X = final_df.iloc[:,1] 
y = final_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("results:")
for (sample, pred) in zip(y_test, y_pred):
    print(sample, ":", pred)
print("accuracy:", accuracy_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(classification_report(y_test, y_pred))
roc = roc_auc_score(y_test, y_pred)
print('roc', roc)


# NER and visuals

# create the corpus of sentences
corpus = []
for i in final_df.index:
    sentence = re.sub('[^a-zA-z]', ' ', final_df.text[i])
    tokens = nlp(sentence)
    tokens = [t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_ for t in tokens]
    tokens = [t for t in tokens if t not in set(list(ENGLISH_STOP_WORDS)+list(punctuation)+ [
            "n't","'d","'s", "'m", "--", "---", "...", "“", "”", "'ve"])]  
    tokens = [t for t in tokens if len(t)>2]
    tokens = ' '.join(tokens)
    corpus.append(tokens)

# find most common words that are used in the dataset
from collections import Counter

flat_list = []
for i in range(0,len(corpus)):
    flat_list.append(corpus[i].split())
flat_list = [item for sublist in flat_list for item in sublist]

print('tokens: ',flat_list[:5])
print()
# first ten
print('top 10 most frequent words: ')
Counter(flat_list).most_common(10)


# find name entities and append to a list of tuples.
def ner(sent):
    ent = nlp(sent)
    ent = [(e.text, e.label_) for e in ent.ents ]
    return(ent)

ents = []

for i in final_df.index:
    ents.append(ner(final_df.text[i]))

ent_list = [item for sublist in ents for item in sublist]

# delete some whitespaces that wrongly appended to the list
for i in range(0,len(ent_list)):
    try:
        if ent_list[i][0] == '  ' or ent_list[i][0] == ' ':
            del ent_list[i]
    except IndexError:
        pass

print('  word',',','entity',',', 'count')
Counter(ent_list).most_common(30)


# print only Organisations, Persons and Geogrphical entities
for i in range(0,100):
    if ent_list[i][1] == 'ORG':
        print(ent_list[i][0])
    elif ent_list[i][1] == 'PERSON':
        print(ent_list[i][0])
    elif ent_list[i][1] == 'GPE':
        print(ent_list[i][0])

for i,j in enumerate(Counter(ent_list).most_common(50)):
    if Counter(ent_list).most_common(50)[i][0][1] == 'ORG':
        print(Counter(ent_list).most_common(50)[i][0][0], 
              Counter(ent_list).most_common(50)[i][1])
    elif Counter(ent_list).most_common(50)[i][0][1] == 'PERSON':
        print(Counter(ent_list).most_common(50)[i][0][0], 
              Counter(ent_list).most_common(50)[i][1])
    elif Counter(ent_list).most_common(50)[i][0][1] == 'GPE':
        print(Counter(ent_list).most_common(50)[i][0][0], 
              Counter(ent_list).most_common(50)[i][1])        

# need only the text from entities for 'NER word cloud'
def ner2(sent):
    ent = nlp(sent)
    ent = [e.text for e in ent.ents ]
    return(ent)

words4cloud = []

for i in final_df.index:
    words4cloud.append(ner2(final_df.text[i]))

words4cloud = [item for sublist in words4cloud for item in sublist]

# create a single string from the list's content
words4cloud = ' '.join(words4cloud)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud().generate(words4cloud)

plt.figure(figsize = (12, 10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.title('Most Frequent Words', fontsize = 15)
plt.show()

wordcloud = WordCloud(max_font_size = 40).generate(words4cloud)
plt.figure(figsize = (12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Most Frequent Words / fontsize limit', fontsize = 15)
plt.axis("off")
plt.show()


# only names of persons included
persons = []
for i,j in enumerate(ent_list):
    if ent_list[i][1] == 'PERSON':
        persons.append(ent_list[i][0])
persons = ' '.join(persons)

wordcloud_p = WordCloud().generate(persons)

plt.figure(figsize = (12, 10))
plt.imshow(wordcloud_p, interpolation = 'bilinear')
plt.title('Most Common names of Persons', fontsize = 15)
plt.axis("off")
plt.show()

# this one is for organisations
orgs = []
for i,j in enumerate(ent_list):
    if ent_list[i][1] == 'ORG':
        orgs.append(ent_list[i][0])
orgs = ' '.join(orgs)

wordcloud_or = WordCloud().generate(orgs)

plt.figure(figsize = (12, 10))
plt.imshow(wordcloud_or, interpolation = 'bilinear')
plt.title('Most Common names of Organisations', fontsize = 15)
plt.axis("off")
plt.show()

