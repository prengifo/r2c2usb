# -*- coding: utf-8 -*-

import re
from danielesis.utils import load_synonyms

def remove_urls(tweet):
    # remove urls
    regexp = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    return re.sub(regexp, '', tweet)

def remove_mentions(tweet):
    regexp = r'@[A-Za-z0-9-_]+'
    return re.sub(regexp, '', tweet)

def remove_safe_html(tweet):
    regexp = r'(&gt;)+'
    return re.sub(regexp, '', tweet)

def remove_accents(tweet):
    # for (search, replace) in zip(u'áéíóúñ'.encode('utf-8'), u'aeioun'):
    #     tweet = tweet.replace(search.decode('utf-8'), replace)

    return tweet


def remove_words(tweet):
    words = set(['via', 'RT', 'rt'])
    split_tweet = [word for word in tweet.lower().split(' ') if word.strip()]
    return ' '.join([word.strip()
                     for word in split_tweet
                     if word not in words and not word.isdigit()])

def fix_errors(tweet):
    dict = load_synonyms('./datasets/sinonimos.csv')
    for key, value in dict.iteritems():
        re.sub(key, value, tweet)
    return tweet


def process_tweet(tweet):
    pipeline = [remove_urls, remove_mentions, remove_safe_html,
                remove_accents, remove_words]
    for func in pipeline:
        tweet = func(tweet)

    return tweet

# def remove_hours(tweet):

#     # # remover horas
#     # regexp = r'[0-9:y]+(am|pm)'
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r'[0-9:y]+'
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r'a. m.'
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r'p. m.'
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r' am '
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r' pm '
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r'km|mtrs'
#     # tweet = re.sub(regexp, '', tweet)

#     # regexp = r'#donatusmedicamentos'
#     # tweet = re.sub(regexp, '', tweet)


#     # remover nombres
#     regexp = r'@[A-Za-z0-9-_]+'
#     tweet = re.sub(regexp, '', tweet)

#     regexp = r'(&gt;)+'
#     tweet = re.sub(regexp, '', tweet)

#     # remover acentos
#     # tweet = ''.join((c for c in unicodedata.normalize('NFD', unicode(tweet.decode('utf-8'))) if unicodedata.category(c) != 'Mn')).encode('utf-8')
#     # for (search, replace) in zip(u'áéíóú', u'aeiou'):
#     #     tweet = tweet.replace(search.decode('utf-8'), replace)

#     regexp = set(['via', 'RT', 'rt'])
#     # remove stop words
#     split_tweet = tweet.lower().split(' ')
#     new_s = []
#     for word in split_tweet:
#         word = word.strip()
#         if word and word not in regexp and not word.isdigit():
#             new_s.append(word)
#     tweet = ' '.join(new_s)
#     return tweet
