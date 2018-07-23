'''
Sentiment Analysis Using Interesting Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import metrics
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.datasets import load_files
from sklearn.externals import joblib
from re import sub
import string
import os

def Run_Classifier(pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, pickle_enable):
    '''    Run Classifier after Preprocessing and Parameters are done    '''
    ## TRAIN
    grid_go = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_go = grid_go.fit(data_train, labels_train)
    print('- - - - - BEST PARAMETERS - - - - -')
    print(grid_go.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, grid_go.best_params_[param_name]))

    print('\n- - - - - DETAILS - - - - -')
    for i in range(len(grid_go.cv_results_['params'])):
        results_noStopWords = copy.deepcopy(grid_go.cv_results_['params'][i])
        results_noStopWords.pop('vect__stop_words')
        print(i, 'params - %s; mean - %0.10f; std - %0.10f'
                    #% (list((grid_go.cv_results_['params'][i]).values())[:-1],  # Print all except the Stopwords
                    % (results_noStopWords.values(),
                    grid_go.cv_results_['mean_test_score'][i],
                    grid_go.cv_results_['std_test_score'][i]))

    # Model Persistence - Pickle
    if pickle_enable == 1: joblib.dump(grid_go.best_estimator_, 'PreTrainedNaiveBayesLexicon.pkl') 

    ## PREDICT
    predicted = grid_go.predict(data_test)

    Print_Result_Metrics(labels_test, predicted, targetnames)  

def Print_Result_Metrics(labels_test, predicted, targetnames):
    '''    Print Metrics after Training etc.    '''
    print('\n- - - - - RESULT METRICS - - - - -')
    print('Exact Accuracy: ', metrics.accuracy_score(labels_test, predicted))
    print(metrics.classification_report(labels_test, predicted, target_names=targetnames))
    print(metrics.confusion_matrix(labels_test, predicted))

class LemmaTokenizer(object):
    '''    Override SciKit's default Tokenizer    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        # This punctuation remover has the best Speed Performance
        self.translator = str.maketrans('','', sub('\'', '', string.punctuation))
    def __call__(self, doc):
        # return [self.wnl.lemmatize(t.lower()) for t in word_tokenize(doc)]
        temp = []
        for t in word_tokenize(doc):
            x = t.translate(self.translator) 
            if x != '': temp.append(self.wnl.lemmatize(x.lower())) 
        
        return temp

### PREPROCESSING
dataset = load_files('./datasets/review_polarity/txt_sentoken', shuffle=False)

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Split, X and y are pairs: data & labels
data_train, data_test, labels_train, labels_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=22)


### LET'S BUILD : SentiWordNet - Counting/Spotting Sentimental Words

#SentiWordNet
#n - NOUN
#v - VERB
#a - ADJECTIVE
#s - ADJECTIVE SATELLITE
#r - ADVERB 

### 1. Bing Liu Word Counting
### 2. Custom Trained Naive Bayes Bag of Words, Word Counting

### 3. Bing Liu SentiWordNet Counting
### 4. CUsom Trained Bag of Words, Word Counting

# two ways to SentiWordNet, with or without Word Disambiguation

# IF I GO FOR 2nGRAMS MIN MAX LIMITS NEED TO BE VERY LIGHT SAME FOR STOPWORDS


### LET'S BUILD : NaiveBayes
pipeline1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

# parameters = {'tfidf__use_idf': (True, False),
#               'vect__max_df': [0.95, 0.85, 0.80],
#               'vect__min_df': [5],  # 5 meaning 5 documents
#               'vect__ngram_range': [(1, 1), (1, 2)],              
#               'vect__stop_words': [stopwords.words("english"), 'english', stopwords_complete, stopwords_complete_lemmatized],  # NLTK Stopwords, SciKit Stopwords, Both
#               'vect__strip_accents': ['unicode', None],
#               'vect__tokenizer': [LemmaTokenizer(), None],}

# Optimal
parameters = {'tfidf__use_idf': [True],
              'vect__max_df': [0.80],
              'vect__min_df': [5],
              'vect__ngram_range': [(1, 1)],              
              'vect__stop_words': [stopwords_complete_lemmatized],
              'vect__strip_accents': ['unicode'],
              'vect__tokenizer': [LemmaTokenizer()],}

#Run_Classifier(pipeline1, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, 0)


### LET'S BUILD : SGDC-SVM
pipeline2 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=None, n_jobs=-1)),])

# parameters = {'clf__alpha': [1e-4, 1e-3, 1e-2],
#               'tfidf__use_idf': (True, False),
#               'vect__max_df': [0.95, 0.85, 0.80],
#               'vect__min_df': [5],  # 5 meaning 5 documents
#               'vect__ngram_range': [(1, 1), (1, 2)],              
#               'vect__stop_words': [stopwords.words("english"), 'english', stopwords_complete, stopwords_complete_lemmatized],  # NLTK Stopwords, SciKit Stopwords
#               'vect__strip_accents': ['unicode', None],
#               'vect__tokenizer': [LemmaTokenizer(), None],}

# Optimal
parameters = {'clf__alpha': [1e-3],
              'tfidf__use_idf': [True],
              'vect__max_df': [0.80],
              'vect__min_df': [5],
              'vect__ngram_range': [(1, 1)],              
              'vect__stop_words': [stopwords_complete_lemmatized],
              'vect__strip_accents': ['unicode'],
              'vect__tokenizer': [LemmaTokenizer()],}

#Run_Classifier(pipeline2, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, 0)


## LET'S BUILD : SVM
pipeline3 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC(loss='hinge', penalty='l2', max_iter=1000, dual=True)),]) # dual for Text Classification

# parameters = {'clf__C': [1, 500, 1000],
#               'tfidf__use_idf': (True, False),
#               'vect__max_df': [0.95, 0.85, 0.80],
#               'vect__min_df': [5],  # 5 meaning 5 documents
#               'vect__ngram_range': [(1, 1), (1, 2)],              
#               'vect__stop_words': [stopwords.words("english"), 'english', stopwords_complete, stopwords_complete_lemmatized],  # NLTK Stopwords, SciKit Stopwords
#               'vect__strip_accents': ['unicode', None],
#               'vect__tokenizer': [LemmaTokenizer(), None],}

# Optimal
parameters = {'clf__C': [500],
              'tfidf__use_idf': [True],
              'vect__max_df': [0.80],
              'vect__min_df': [5],
              'vect__ngram_range': [(1, 1)],              
              'vect__stop_words': [stopwords_complete_lemmatized],
              'vect__strip_accents': ['unicode'],
              'vect__tokenizer': [LemmaTokenizer()],}

#Run_Classifier(pipeline3, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, 0)