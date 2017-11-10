# -*- coding: utf8 -*-

import codecs, os, io
import re, math
from pyvi.pyvi import ViTokenizer
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import numpy as np

folder_train = '/Volumes/DATA/workspace/vnu/spc/git/doit-classification/DataSource/Data_raw/train'
folder_test = '/Volumes/DATA/workspace/vnu/spc/git/doit-classification/DataSource/Data_raw/test'

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if filepath.find('.txt') != -1:
                file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def load_dataset(folder):
    dataset = {'target_names': [], 'data': [], 'target': []}
    print('loading dataset')
    for root, dirs, files in os.walk(folder, topdown=False):
        position = 0
        for name in dirs:
            subdir = os.path.join(root, name)
            dataset['target_names'].append(name)
            filesPath = get_filepaths(subdir)
            for filePath in filesPath:
                with io.open(filePath, mode="r", encoding="UTF8") as file:
                    content = file.read().lower()
                    rx = re.compile("[^\W\d_]+", re.UNICODE)
                    content = " ".join(rx.findall(content))
                    dataset['data'].append(ViTokenizer.tokenize(content))
                    dataset['target'].append(position)
            position += 1

    return dataset, dataset['target_names']


def train():
    train_set, target_names = load_dataset(folder_train)
    joblib.dump(train_set, 'train_set.pkl')
    print('loaded train_set')

    # Extracting features from text files
    from sklearn.feature_extraction.text import CountVectorizer

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_set['data'])
    joblib.dump(X_train_counts, 'X_train_counts.pkl')
    print(X_train_counts.shape)

    # TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    joblib.dump(X_train_tfidf, 'X_train_tfidf.pkl')
    print('tfidf')
    print(X_train_tfidf.shape)

    # Training Naive Bayes (NB) classifier on training data.
    # from sklearn.naive_bayes import MultinomialNB
    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    print('training')
    # text_clf = text_clf.fit(train_set['data'], train_set['target'])
    # joblib.dump(text_clf, 'text_clf.pkl')

    # In[16]:
    print('svm')
    # Training Support Vector Machines - SVM and calculating its performance

    from sklearn.linear_model import SGDClassifier

    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm',
                              SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    text_clf_svm = text_clf_svm.fit(train_set['data'], train_set['target'])
    joblib.dump(text_clf_svm, 'text_clf_svm.pkl')
    return target_names

def test():
    # test_set = load_dataset(folder_test)
    # joblib.dump(test_set, 'test_set.pkl')
    print('process test')
    test_set = joblib.load('test_set.pkl')
    print('loaded test_set')
    text_clf_svm = joblib.load('text_clf_svm.pkl')
    predicted_svm = text_clf_svm.predict(test_set['data'])
    print(np.mean(predicted_svm == test_set['target']))


def load_text(doc):
    dataset = {'target_names': [], 'data': [], 'target': []}
    content = doc.lower()
    rx = re.compile("[^\W\d_]+", re.UNICODE)
    content = " ".join(rx.findall(content))
    dataset['data'].append(ViTokenizer.tokenize(content))
    return dataset

def predict(predict_set):
    print('process predict')
    text_clf_svm = joblib.load('text_clf_svm.pkl')
    # predicted_svm = text_clf_svm.predict_proba(predict_set['data'])
    predicted_svm = text_clf_svm.predict(predict_set['data'])
    # print(predicted_svm)
    target_names = joblib.load('train_set.pkl')['target_names']
    index_classification = math.floor(np.mean(predicted_svm))
    return index_classification, target_names[index_classification]

def load_file(filePath):
    with io.open(filePath, mode="r", encoding="UTF8") as file:
        return file.read().lower()



