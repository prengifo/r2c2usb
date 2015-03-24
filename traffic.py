# coding: utf-8
import csv
import unicodedata
import random
import re
import numpy as np
import scipy
import itertools
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.learning_curve import learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import cross_validation, preprocessing, grid_search
import networkx as nx

import matplotlib.pyplot as plt
from termcolor import colored, cprint
# from .retriever import relevant_tweets, relevant_tweets_time

def word_matrix(corpus):
    # print '\n'.join(x.decode('utf-8') for x in corpus).encode('utf-8')
    vectorizer = TfidfVectorizer(min_df=1# , ngram_range=(1, 2),
                                 # token_pattern=r'\b\w+\b'
    )
    # vectorizer = CountVectorizer(min_df=1# , ngram_range=(1, 2),
    #                              # token_pattern=r'\b\w+\b'
    # )

    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names()
    return vectorizer, X, vocab

def process_tweet(tweet):
    # remove urls
    regexp = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    tweet = re.sub(regexp, '', tweet)

    # # remover horas
    # regexp = r'[0-9:y]+(am|pm)'
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r'[0-9:y]+'
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r'a. m.'
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r'p. m.'
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r' am '
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r' pm '
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r'km|mtrs'
    # tweet = re.sub(regexp, '', tweet)

    # regexp = r'#donatusmedicamentos'
    # tweet = re.sub(regexp, '', tweet)


    # remover nombres
    regexp = r'@[A-Za-z0-9-_]+'
    tweet = re.sub(regexp, '', tweet)

    regexp = r'(&gt;)+'
    tweet = re.sub(regexp, '', tweet)

    # remover acentos
    # tweet = ''.join((c for c in unicodedata.normalize('NFD', unicode(tweet.decode('utf-8'))) if unicodedata.category(c) != 'Mn')).encode('utf-8')
    # for (search, replace) in zip(u'áéíóú', u'aeiou'):
    #     tweet = tweet.replace(search.decode('utf-8'), replace)

    regexp = set(['via', 'RT', 'rt'])
    # remove stop words
    split_tweet = tweet.lower().split(' ')
    new_s = []
    for word in split_tweet:
        word = word.strip()
        if word and word not in regexp and not word.isdigit():
            new_s.append(word)
    tweet = ' '.join(new_s)
    return tweet


def load_labels(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            row[0] = process_tweet(row[0])

            # try:
            row[1] = int(row[1])
            # except ValueError:
            #     pass

            if row[0].strip():
                rows.append(row)
    return rows

def load_labels2(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            row[0] = process_tweet(row[0])

            # try:
            row[1] = int(row[1])
            # except ValueError:
            #     pass

            if row[0].strip():
                rows.append(row)
    return rows


def get_tweets_text(rows):
    return [x[0] for x in rows]

def get_relevant_labels(rows):
    return [x[1] for x in rows]

def get_traffic_labels(rows):
    # return [x[2] for x in rows]
    return [x[1] for x in rows]

def get_dir_labels(rows):
    from sklearn.preprocessing import MultiLabelBinarizer
    return MultiLabelBinarizer().fit_transform(
        [[int(y) for y in x[1].split(',') if y] for x in rows]
    )

def filter_for_traffic(rows):
    tt = []
    for x in rows:
        if x[1] == 1 and x[3] != '':
            tt.append(x)
    return tt

def train_model(clf, filename, filter_func=lambda x: x,
                cross_validate=False, validate=False, validate_filename='p2.csv',
                test_size=0.2, random_state=None,
                extract_labels=get_relevant_labels, ll=load_labels):
    rows = filter_func(ll(filename))
    Y = extract_labels(rows)
    # print Y.
    vectorizer, X, vocab = word_matrix(get_tweets_text(rows))
    print vocab
    # X = preprocessing.scale(X, with_mean=False)
    # else:
    #     vectorizer, X, vocab = use


    # svd = TruncatedSVD(1000)
    # lsa = make_pipeline(svd, Normalizer(copy=False))

    # X = svd.fit_transform(X)
    # print X.shape

    if cross_validate:
        title = "Learning Curves (Logistic Regression)"
        cv = cross_validation.ShuffleSplit(X.shape[0],
                                           n_iter=100,
                                           test_size=test_size)
        plot_learning_curve(clf, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        plt.show()
        # scores = cross_validation.cross_val_score(clf, X, Y,
        #                                           cv=cv)
        # # print scores
        # print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

    else:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y,
                                                                             test_size=test_size,
                                                                             random_state=None)


        clf.fit(X_train, y_train)
        print 'accuracy on testing set', clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        # print 'TEST REAL'
        # print y_test
        # print 'TEST PRED'
        # print y_pred

        # for x,y in zip(y_test, y_pred):
        #     print x,y

        cm = confusion_matrix(y_test, y_pred)
        print cm
        print classification_report(y_test, y_pred, target_names=['verde', 'amarillo', 'rojo'])
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        if validate:
            rows_validate = filter_func(ll(validate_filename))
            X_validation = vectorizer.transform(get_tweets_text(rows_validate))
            y_validation = extract_labels(rows_validate)
            print 'accuracy on validation set', clf.score(X_validation,
                                                          y_validation)

        return clf, vectorizer, X, vocab

def get_relevant(**kwargs):
    # clf = kwargs.pop('clf', LogisticRegression())
    clf = kwargs.pop('clf', LogisticRegression(C=10))
    return train_model(clf, 'relevant.csv', **kwargs)

def get_traffic(**kwargs):
    clf = kwargs.pop('clf', LogisticRegression(C=10))
    # clf = kwargs.pop('clf', OneVsRestClassifier(svm.LinearSVC())) # 0.7 +- 0.11
    # parameters = [{'C': [0.2] + [2**x for x in range(-10, 11)]}]
    # clf = grid_search.GridSearchCV(svm.LinearSVC(), parameters, n_jobs=4)
    # clf = grid_search.RandomizedSearchCV(svm.LinearSVC(), parameters, n_iter=100)
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.2)) # 0.7 +- 0.11
    # clf = kwargs.pop('clf', MultinomialNB()) # 0.7 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(multi_class='crammer_singer', C=0.3)) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.09)) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.24)) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.256)) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.2)) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.25, class_weight='auto')) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.LinearSVC(C=0.25, class_weight='auto')) # 0.69 +- 0.11
    # clf = kwargs.pop('clf', svm.SVC()) # 0.69 +- 0.11
    model = train_model(clf, 'traffic.csv', **kwargs)
    # print clf.get_params()
    return model

# def get_dir(**kwargs):
#     # clf = kwargs.pop('clf', LogisticRegression(C=10))
#     # clf = kwargs.pop('clf', OneVsRestClassifier(LogisticRegression())) # 0.7 +- 0.11
#     clf = kwargs.pop('clf', OneVsRestClassifier(svm.LinearSVC(C=10))) # 0.7 +- 0.11
#     # clf = kwargs.pop('clf', svm.LinearSVC(C=0.25, class_weight='auto')) # 0.69 +- 0.11
#     model = train_mode-l(clf, 'leDir.csv', extract_labels=get_dir_labels,
#                         ll=load_labels2, **kwargs)
#     return model

def color_code_text(text, score):
    COLORS = ['green', 'yellow', 'red']
    cprint(text, COLORS[score])

def predict1(estimator, vectorizer, tweet):
    # print 'ACA!!!!!'
    # print vectorizer.transform([tweet])
    # print estimator.predict(vectorizer.transform([tweet]))
    return estimator.predict(vectorizer.transform([tweet]))[0]

# def partition_historical(via, window=30, since_date=None):
#     relevant, relevant_v, relevant_X, relevant_vocab = get_relevant(test_size=0.3)
#     traffic, traffic_v, traffic_X, traffic_vocab = get_traffic(test_size=0.3)
#     # window debe dividir a 60 minutos (1 hora)
#     scores = defaultdict(lambda: [0, 0, 0])
#     for hour in range(24):
#         for minutes in range(0, 60, window):
#             start = '%02d:%02d:00' % (hour, minutes)
#             end = '%02d:%02d:59' % (hour, (minutes + window - 1) % 60)
#             print start, end
#             for tweet in relevant_tweets_time(via, start, end, since_date):
#                 if predict1(relevant, relevant_v, tweet.text) == 1:
#                     predicted_score = predict1(traffic, traffic_v, tweet.text)
#                     color_code_text(tweet.text.encode('utf-8'), predicted_score)
#                     scores[(start, end)][predicted_score] += 1
#     return scores

def color_code():
    from lazy import AYER
    relevant, relevant_v, relevant_X, relevant_vocab = get_relevant(test_size=0.3)
    traffic, traffic_v, traffic_X, traffic_vocab = get_traffic(test_size=0.3)
    # print AYER
    for key, tweets in AYER.items(): # itertools.groupby(relevant_tweets('trinidad', 25*60), key=lambda x: x.created_at.hour):
        # print key, list([x.text.encode('utf-8') for x in tweets])
        s,n = 0, 0
        print 'Desde las {}:00 hasta las {}:59'.format(key, key)
        for x in tweets:
            x = unicode(x.decode('utf-8'))
            # print x
            if predict1(relevant, relevant_v, x) == 1:
                # print 'aca'
                s += predict1(traffic, traffic_v, x)
                n += 1
                if predict1(traffic, traffic_v, x) == 0:
                    cprint(x.encode('utf-8'), 'green')
                elif predict1(traffic, traffic_v, x) == 1:
                    cprint(x.encode('utf-8'), 'yellow')
                elif predict1(traffic, traffic_v, x) == 2:
                    cprint(x.encode('utf-8'), 'red')
        print "Promedio de cola", (s/float(n) if n > 0 else 0)

# from datetime import datetime, timedelta
# ayer = datetime.now().date() - timedelta(days=30)
# scores = partition_historical('panamericana', window=60, since_date=ayer)
# with open('maldita-pnm.csv', 'w') as f:
#     writer = csv.writer(f)
#     for time_range in sorted(scores.keys()):
#         # total = sum(scores[time_range])
#         # weighted = sum(i*x for i, x in enumerate(scores[time_range]))
#         print time_range, scores[time_range], weighted, weighted/float(total)
#         writer.writerow(['pnm', time_range[0], time_range[1]] + scores[time_range])

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    # get_relevant(test_size=0.4, cross_validate=True)
    # get_relevant(test_size=0.4)
    # get_dir(test_size=0.2)
    # get_traffic(test_size=0.2) # , cross_validate=True)
    color_code()
    G=nx.Graph()
    G.add_nodes_from(['El Cafetal','USB','Los Naranjos','El Hatillo','El Volcan','El Placer','Piedra Azul','Hoyo de la Puerta','La Trinidad','Los Samanes','Vizcaya','Los Ruices','Distribuidor AFF','La Rinconada'])

    G.add_edges_from([('El Cafetal','Los Naranjos'),('El Cafetal','Vizcaya'),('El Cafetal','Los Ruices'),('Los Naranjos','El Hatillo'),('El Hatillo','El Volcan'),('El Volcan','El Placer'),('El Placer','USB'),('Piedra Azul','El Placer'),('La Trinidad','Piedra Azul'),('El Hatillo','La Trinidad'),('Vizcaya','Los Samanes'),('Los Samanes','La Trinidad'),('Los Ruices','Distribuidor AFF'),('Distribuidor AFF','La Trinidad'),('Distribuidor AFF','La Rinconada'),('La Rinconada','Hoyo de la Puerta'),('Hoyo de la Puerta','USB')])

    G['El Cafetal']['Los Ruices']['rutas']= ['San Luis','Santa Paula','Santa Marta','Santa Sofia','Cafetal','Bulevard','Raul Leoni']

    G['El Cafetal']['Vizcaya']['rutas'] = ['Vizcaya','La Guairita']

    G['El Cafetal']['Los Naranjos']['rutas'] = ['Plaza las Americas','Los Naranjos']

    G['Los Naranjos']['El Hatillo']['rutas'] = ['Hatillo','Universidad Nueva Esparta','La Muralla','Carretera vieja tocuyito']

    G['El Hatillo']['El Volcan']['rutas'] = ['Kavak','Volcan']

    G['El Volcan']['El Placer']['rutas'] = ['Oripoto','Gavilan','Jean Piglet']

    G['El Placer']['USB']['rutas'] = ['El Placer','USB']

    G['Hoyo de la Puerta']['USB']['rutas'] = ['Hoyo de la Puerta','USB']

    G['La Rinconada']['Hoyo de la Puerta']['rutas'] = ['Tazon','La Rinconada','Charallave','La Victoria','Valencia','Valles del Tuy','Ocumitos','Las Mayas','ARC']

    G['Distribuidor AFF']['La Rinconada']['rutas'] = ['Valle Coche','El Pulpo','Santa Monica','Proceres','Chaguaramos','La Bandera']

    G['Los Ruices']['Distribuidor AFF']['rutas'] = ['Francisco Fajardo','El Pulpo','La Polar','Santa Cecilia','Distribuidor Altamira','Soto']

    G['Vizcaya']['Los Samanes']['rutas'] = ['Los Samanes','La Guairita']

    G['Los Samanes']['La Trinidad']['rutas'] = ['Los Samanes','La Trinidad','Procter']

    G['Distribuidor AFF']['La Trinidad']['rutas'] = ['Prados del Este','Santa Fe','Concresa','Santa Rosa de Lima','Ciempies','Valle Arriba','Terrazas del Club','Los Campitos','Tunel de la Trinidad']

    G['La Trinidad']['Piedra Azul']['rutas'] = ['Baruta','EPA','La Trinidad','Expreso']

    G['Piedra Azul']['El Placer']['rutas'] = ['El Placer','Los Guayabitos','Ojo de Agua','Monterrey']

    G['La Trinidad']['El Hatillo']['rutas'] = ['La Trinidad','El Hatillo','La Boyera']

    print G.edges(data=True)
