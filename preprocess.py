# -*- coding: utf-8 -*-

import re, unicodedata
from nltk.stem.snowball import SpanishStemmer
# from whoosh.lang.porter import stem
from danielesis.spelling import correct
def remove_urls(tweet):
    # remove urls
    regexp = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return re.sub(regexp, '', tweet)

def remove_mentions(tweet):
    regexp = r'@[A-Za-z0-9-_]+'
    return re.sub(regexp, '', tweet)

def remove_punctuation(tweet):
    to_be_removed = ".,:!?()-"
    for c in to_be_removed:
        tweet = tweet.replace(c, ' ')
    return tweet

def remove_safe_html(tweet):
    regexp = r'(&gt;)+'
    return re.sub(regexp, '', tweet)

def remove_accents(tweet):
    tweet = unicode(tweet, "utf-8")
    return unicodedata.normalize('NFKD', tweet).encode('ASCII', 'ignore')

def remove_hashtag(tweet):
    regexp = r'#donatusmedicamentos'
    tweet = re.sub(regexp, '', tweet)
    return tweet

def remove_words(tweet):
    words = set(['via', 'RT', 'rt'])
    split_tweet = [word for word in tweet.lower().split(' ') if word.strip()]
    return ' '.join([word.strip()
                     for word in split_tweet
                     if word not in words and not word.isdigit()])

def fix_errors(tweet):
    split_tweet = [word for word in tweet.lower().split(' ') if word.strip()]
    return ' '.join([correct(word.strip())
                     for word in split_tweet])

def grammar_fix(tweet):
    for key, value in dict.iteritems():
        re.sub(key, value, tweet)
    return correct(tweet)

def stemmer_all(tweet):
    stm = SpanishStemmer()
    split_tweet = [word for word in tweet.lower().split(' ') if word.strip()]
    return ' '.join([stm.stem(word.strip())
                     for word in split_tweet])


def process_tweet(tweet):
    pipeline = [remove_urls, remove_mentions, remove_safe_html,
                remove_accents, remove_hashtag, remove_punctuation, fix_errors,
                stemmer_all, remove_words, grammar_fix]

    # pipeline = []
    # print 'old %s' % tweet
    for func in pipeline:
        tweet = func(tweet)
    # print 'new %s' % tweet
    return tweet
