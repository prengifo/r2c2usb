# -*- coding: utf-8 -*-
import rfc822
import csv

from calendar import timegm
from datetime import datetime
from termcolor import cprint


def parse_datetime(obj):
    return datetime.fromtimestamp(timegm(rfc822.parsedate(obj.created_at)))

def color_code_text(text, score):
    COLORS = ['green', 'yellow', 'red']
    cprint(text, COLORS[score])

def load_file(filename):
    '''Loads a csv file, returns a freaking generator because yes'''
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

def load_tweets(filename):
    '''Only load tweets, no labels'''
    return (x[0] for x in load_file(filename))

def write_file(filename, rows):
    '''Write `rows` to `filename`'''
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def load_synonyms(filename):
    '''Loads synonyms in a dictionary'''
    synonyms = {}
    for x in load_file(filename):
        synonyms.update({x[0].decode('iso-8859-1').encode('utf8'):x[1]})
    return synonyms

def load_words():
    '''Loads the words for the corrector'''
    s = set()
    for x in load_file('./datasets/lemario-general-del-espanol.txt'):
        s.add(x[0])
    for x in load_file('./datasets/verbos-espanol.txt'):
        s.add(x[0])
    for x in load_file('./datasets/verbos-espanol-conjugaciones.txt'):
        s.add(x[0])
    for x in load_file('./datasets/nombres-propios-es.txt'):
        s.add(x[0])
    for x in load_file('./datasets/apellidos-es.txt'):
        s.add(x[0])
    return s