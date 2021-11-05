# Michael Prappas
# EMIS 8331 - Adv. Data Mining
# Final Project

import nltk
from nltk.sentiment import SentimentAnalyzer
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
from nltk import tokenize
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#
# GET DATA LOADED
#

# get CSV tools
# fields are huge, so have to increase size
import csv
csv.field_size_limit(262144)

with open('fake_or_real_news.csv', 'rb') as f:
    reader = csv.reader(f)
    news = list(reader)

news[3][2]
len(news)
# 6336 (one header row - 6335 articles)
len(news[0])
# 4

#
# feature extraction using count vectorizer
# preprocessing to replace specific numbers with generic number
# preprocessing in vectorizer: remove stopwords, strip accents
#

# get just the article data
just_articles = [row[2] for row in news]
# remove first article
just_articles.pop(0)

# replace all numbers with the number 0
just_articles = [re.sub(r'\d+', '0', article) for article in just_articles]

vectorizer = CountVectorizer(input = just_articles,
                             decode_error = 'replace',
                             strip_accents = 'ascii',
                             analyzer = 'word',
                             ngram_range = (2, 2),
                             stop_words = 'english',
                             lowercase = True,
                             min_df = 3)

article_term_matrix = vectorizer.fit_transform(just_articles)
vocabulary = vectorizer.get_feature_names()
# 122,930 unique bigrams

#
# get k-best features using Chi-squared test
# 122,930 features total
# increments of 1X and 5X features
#

# just get the labels (fake/real)
just_labels = [row[3] for row in news]
# remove header
just_labels.pop(0)

chi_sq_100   = SelectKBest(chi2, k =   100)
chi_sq_500   = SelectKBest(chi2, k =   500)
chi_sq_1000  = SelectKBest(chi2, k =  1000)
chi_sq_5000  = SelectKBest(chi2, k =  5000)
chi_sq_10000 = SelectKBest(chi2, k = 10000)
chi_sq_50000 = SelectKBest(chi2, k = 50000)

# 100
article_term_matrix_100 = chi_sq_100.fit_transform(article_term_matrix, just_labels)
feature_indices_100 = chi_sq_100.get_support(indices = True)
vocabulary_sel_100 = []
for index in feature_indices_100:
    vocabulary_sel_100.extend([vocabulary[index]])

# 500
article_term_matrix_500 = chi_sq_500.fit_transform(article_term_matrix, just_labels)
feature_indices_500 = chi_sq_500.get_support(indices = True)
vocabulary_sel_500 = []
for index in feature_indices_500:
    vocabulary_sel_500.extend([vocabulary[index]])

# 1000
article_term_matrix_1000 = chi_sq_1000.fit_transform(article_term_matrix, just_labels)
feature_indices_1000 = chi_sq_1000.get_support(indices = True)
vocabulary_sel_1000 = []
for index in feature_indices_1000:
    vocabulary_sel_1000.extend([vocabulary[index]])

# 5000
article_term_matrix_5000 = chi_sq_5000.fit_transform(article_term_matrix, just_labels)
feature_indices_5000 = chi_sq_5000.get_support(indices = True)
vocabulary_sel_5000 = []
for index in feature_indices_5000:
    vocabulary_sel_5000.extend([vocabulary[index]])

# 10000
article_term_matrix_10000 = chi_sq_10000.fit_transform(article_term_matrix, just_labels)
feature_indices_10000 = chi_sq_10000.get_support(indices = True)
vocabulary_sel_10000 = []
for index in feature_indices_10000:
    vocabulary_sel_100.extend([vocabulary[index]])

# 50000
article_term_matrix_50000 = chi_sq_50000.fit_transform(article_term_matrix, just_labels)
feature_indices_50000 = chi_sq_50000.get_support(indices = True)
vocabulary_sel_50000 = []
for index in feature_indices_50000:
    vocabulary_sel_50000.extend([vocabulary[index]])


#
# determine best k to use (number of features)
# use a Naive Bayes (Bernoulli - binary data) classifier
#

clasf = BernoulliNB(alpha = 0.1)
just_labels_binary = []
for label in just_labels:
    if (label == 'FAKE'):
        just_labels_binary.extend([0])
    else:
        just_labels_binary.extend([1])

# 100
scores = cross_val_score(clasf, article_term_matrix_100, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.79071589387639885
scores = cross_val_score(clasf, article_term_matrix_100, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.88191256832149845

# 500
scores = cross_val_score(clasf, article_term_matrix_500, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.83174984506395044
scores = cross_val_score(clasf, article_term_matrix_500, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.93558021724963081

# 1000
scores = cross_val_score(clasf, article_term_matrix_1000, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.84784890653371736
scores = cross_val_score(clasf, article_term_matrix_1000, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.94883386958178928

# 5000
scores = cross_val_score(clasf, article_term_matrix_5000, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.89757113823718293
scores = cross_val_score(clasf, article_term_matrix_5000, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.97487510985697556

# 10000
scores = cross_val_score(clasf, article_term_matrix_10000, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.90924906156849727
scores = cross_val_score(clasf, article_term_matrix_10000, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.98235360115624837

# 50000
scores = cross_val_score(clasf, article_term_matrix_50000, just_labels, cv = 10, scoring = 'accuracy')
accuracy = sum(scores) / len(scores)
accuracy
# accuracy: 0.95201776994956211
scores = cross_val_score(clasf, article_term_matrix_50000, just_labels_binary, cv = 10, scoring = 'roc_auc')
ROC_AUC = sum(scores) / len(scores)
ROC_AUC
# ROC_AUC: 0.99272630415209284

# to eliminate overfitting, approximate starting point to McIntire's classifier, and have room for improvement,
# using 500 features

article_terms_500 = article_term_matrix_500.toarray().tolist()

with open('article_terms_500.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(article_terms_500)

with open('vocabulary_sel_500.csv', 'wb') as f:
    writer = csv.writer(f)
    for term in vocabulary_sel_500:
        writer.writerow([term])

#
# prep for sentiment analysis
# add fields to tables for sentiment features
#

# add fields to news
news[0].extend(['title_comp', 'title_neg', 'title_neu', 'title_pos', 'title_pol', 'title_subj',
                'first_comp', 'first_neg', 'first_neu', 'first_pos', 'first_pol', 'first_subj',
                 'last_comp',  'last_neg',  'last_neu',  'last_pos',  'last_pol',  'last_subj',
                  'avg_comp',   'avg_neg',   'avg_neu',   'avg_pos',   'avg_pol',   'avg_subj',
                               'most_neg',  'most_neu',  'most_pos',  'high_pol',
                                                                       'low_pol', 'most_subj'])

len(news[0])
# 34 columns

#
# sentiment analysis
#

sia = SentimentIntensityAnalyzer()

# len(news[1]) is 4

for art in range(1, len(news)):

    # get title and article, decode from Unicode
    title = news[art][1]
    title = title.decode('utf-8')
    article = news[art][2]
    article = article.decode('utf-8')

    # get and save title scores
    title_scores = sia.polarity_scores(title)
    news[art].extend([title_scores['compound'], title_scores['neg'], title_scores['neu'], title_scores['pos']])
    title_sentm = TextBlob(title).sentiment
    news[art].extend([title_sentm.polarity, title_sentm.subjectivity])

    # tokenize articles into sentences
    article_sents = tokenize.sent_tokenize(article)

    # len(news[1]) is 10

    if (article_sents == []):
        continue

    # get and save scores for first sentence
    scores = sia.polarity_scores(article_sents[0])
    sentence_sentiment = TextBlob(article_sents[0]).sentiment
    news[art].extend([scores['compound'], scores['neg'], scores['neu'], scores['pos'],
                      sentence_sentiment.polarity, sentence_sentiment.subjectivity])

    # len(news[1]) is 16

    # set up variables for looping thru sentences
    most_neg = scores['neg']
    most_neu = scores['neu']
    most_pos = scores['pos']
    high_pol = sentence_sentiment.polarity
    low_pol = sentence_sentiment.polarity
    most_subj = sentence_sentiment.subjectivity
    compound_sum = scores['compound']
    negative_sum = scores['neg']
    neutral_sum = scores['neu']
    positive_sum = scores['pos']
    polarity_sum = sentence_sentiment.polarity
    subjectivity_sum = sentence_sentiment.subjectivity

    # len(news[1]) is still 16

    # get statistics on the rest of the article
    for sent in range(1, len(article_sents)):
        scores = sia.polarity_scores(article_sents[sent])
        sentence_sentiment = TextBlob(article_sents[sent]).sentiment

        if (scores['neg'] > most_neg):
            most_neg = scores['neg']
        if (scores['neu'] > most_neu):
            most_neu = scores['neu']
        if (scores['pos'] > most_pos):
            most_pos = scores['pos']
        if (sentence_sentiment.polarity > high_pol):
            high_pol = sentence_sentiment.polarity
        if (sentence_sentiment.polarity < low_pol):
            low_pol = sentence_sentiment.polarity
        if (sentence_sentiment.subjectivity > most_subj):
            most_subj = sentence_sentiment.subjectivity

        compound_sum     += scores['compound']
        negative_sum     += scores['neg']
        neutral_sum      += scores['neu']
        positive_sum     += scores['pos']
        polarity_sum     += sentence_sentiment.polarity
        subjectivity_sum += sentence_sentiment.subjectivity


    # len(news[1]) is still 16

    # get and save scores for last sentence
    scores = sia.polarity_scores(article_sents[len(article_sents) - 1])
    sentence_sentiment = TextBlob(article_sents[len(article_sents) - 1]).sentiment
    news[art].extend([scores['compound'], scores['neg'], scores['neu'], scores['pos'],
                      sentence_sentiment.polarity, sentence_sentiment.subjectivity])

    # len(news[1]) is 22

    news[art].extend([compound_sum / len(article_sents),
                      negative_sum / len(article_sents),
                      neutral_sum  / len(article_sents),
                      positive_sum / len(article_sents),
                      polarity_sum / len(article_sents),
                      subjectivity_sum / len(article_sents)])

    news[art].extend([most_neg, most_neu, most_pos, high_pol, low_pol, most_subj])
    # len(news[1]) is 34 (all good)

    if (art % 500 == 0):
        print art

with open('news_sentimental.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(news)