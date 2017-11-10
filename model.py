# -*- coding: utf8 -*-

import codecs, os, io
import re, math
from builtins import enumerate

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

    return dataset


def train():
    # train_set = load_dataset(folder_train)
    # joblib.dump(train_set, 'train_set.pkl')
    # print('loaded train_set')
    train_set = joblib.load('train_set.pkl')

    # Extracting features from text files
    from sklearn.feature_extraction.text import CountVectorizer
    # TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer

    # Training Naive Bayes (NB) classifier on training data.
    # from sklearn.naive_bayes import MultinomialNB
    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    #
    # print('training')
    # text_clf = text_clf.fit(train_set['data'], train_set['target'])
    # joblib.dump(text_clf, 'text_clf.pkl')

    # In[16]:
    print('svm')
    # Training Support Vector Machines - SVM and calculating its performance

    from sklearn.linear_model import SGDClassifier

    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm',
                              SGDClassifier(loss='modified_huber', penalty='l2', n_jobs=-1, alpha=1e-3, n_iter=5,
                                            random_state=42))])

    text_clf_svm = text_clf_svm.fit(train_set['data'], train_set['target'])
    joblib.dump(text_clf_svm, 'text_clf_svm.pkl')
    print('train success')


def test():
    print('process test')
    test_set = load_dataset(folder_test)
    joblib.dump(test_set, 'test_set.pkl')
    # test_set = joblib.load('test_set.pkl')
    print('loaded test_set')
    text_clf_svm = joblib.load('text_clf_svm.pkl')
    # text_clf_svm = joblib.load('text_clf.pkl')  # bayes
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
    predicted_svm = text_clf_svm.predict_proba(predict_set['data'])
    # print(predicted_svm)
    target_names = joblib.load('train_set.pkl')['target_names']

    # index_classification = math.floor(np.mean(predicted_svm))
    names_predict_output = []
    # print(predicted_svm[0][0])

    for index in range(len(predicted_svm[0])):
        names_predict_output.append({'index': index, 'score': predicted_svm[0][index], 'name': target_names[index]})
    print(names_predict_output)




def load_file(filePath):
    with io.open(filePath, mode="r", encoding="UTF8") as file:
        return file.read().lower()

# train()

# test()

doc = load_file(
    '/Volumes/DATA/workspace/vnu/spc/git/doit-classification/DataSource/Data_raw/train/KHTN/01050002682.txt')
# print(doc)
predict_set = load_text(doc)
predict(predict_set)
# index_classification, target_name = predict(predict_set)
# print(index_classification)
# print(target_name)
