import codecs, os,io
from pyvi.pyvi import ViTokenizer
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import numpy as np

folder_train = '/Volumes/DATA/SPC/doit_phase2/git/Classification/Data/DoiT/Data_raw/train'
folder_test = '/Volumes/DATA/SPC/doit_phase2/git/Classification/Data/DoiT/Data_raw/test'



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
                    # regex only az, lowercase
                    # predict lang
                    # remove stopword
                    dataset['data'].append(ViTokenizer.tokenize(file.read()))
                    dataset['target'].append(position)

            position +=1
    return dataset

train_set = load_dataset(folder_train)
joblib.dump(train_set, 'train_set.pkl')
print 'loaded train_set'

test_set = load_dataset(folder_test)
joblib.dump(test_set, 'test_set.pkl')
print 'loaded test_set'



# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_set['data'])
joblib.dump(X_train_counts, 'X_train_counts.pkl')
print X_train_counts.shape

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
joblib.dump(X_train_tfidf, 'X_train_tfidf.pkl')
print 'tfidf'
print X_train_tfidf.shape

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

print 'training'
text_clf = text_clf.fit(train_set['data'], train_set['target'])
joblib.dump(text_clf, 'text_clf.pkl')


# In[16]:
print 'svm'
# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(train_set['data'], train_set['target'])
joblib.dump(text_clf_svm, 'text_clf_svm.pkl')
predicted_svm = text_clf_svm.predict(test_set['data'])
print np.mean(predicted_svm == test_set['target'])

# # In[18]:
#
# # Grid Search
# # Here, we are creating a list of parameters for which we would like to do performance tuning.
# # All the parameters name start with the classifier name (remember the arbitrary name we gave).
# # E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.
#
# from sklearn.model_selection import GridSearchCV
#
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
#
# # In[19]:
#
# # Next, we create an instance of the grid search by passing the classifier, parameters
# # and n_jobs=-1 which tells to use multiple cores from user machine.
# print 'grid search'
# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(train_set['data'], train_set['target'])
#
# # In[23]:
#
# # To see the best mean score and the params, run the following code
#
# print gs_clf.best_score_
# print gs_clf.best_params_
#
#
#
# # In[24]:
#
# # Similarly doing grid search for SVM
# from sklearn.model_selection import GridSearchCV
#
# parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
#                   'clf-svm__alpha': (1e-2, 1e-3)}
#
# gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
# print 'gird search for svm'
# gs_clf_svm = gs_clf_svm.fit(train_set['data'], train_set['target'])
#
# print gs_clf_svm.best_score_
# print gs_clf_svm.best_params_