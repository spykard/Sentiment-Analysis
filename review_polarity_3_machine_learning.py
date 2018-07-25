'''
Sentiment Analysis Using Interesting Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.datasets import load_files
from sklearn.externals import joblib
from re import sub
import numpy as np
import string
import copy

def Run_Classifier(grid_search_enable, pickle_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized):
    '''    Run Classifier with or without Grid Search after Preprocessing is done    '''

    ## PREPARE ON - Grid Search to Look for the Best Parameters
    if grid_search_enable == 1:

        # (1) TRAIN
        grid_go = GridSearchCV(pipeline, parameters, n_jobs=-1)
        grid_go = grid_go.fit(data_train, labels_train)
        print('- - - - - BEST PARAMETERS - - - - -')
        print(grid_go.best_score_, 'Accuracy')
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, grid_go.best_params_[param_name]))

        print('\n- - - - - DETAILS - - - - -')
        for i in range(len(grid_go.cv_results_['params'])):
            results_noStopWords = copy.deepcopy(grid_go.cv_results_['params'][i])
            if results_noStopWords['union__vect1__stop_words'] is not None:  # Don't Print the list of Stopwords
                results_noStopWords['union__vect1__stop_words'] = ['ListOfStopWords']   
            if results_noStopWords['union__vect2__stop_words'] is not None:
                results_noStopWords['union__vect2__stop_words'] = ['ListOfStopWords']           
            print(i, 'params - %s; mean - %0.10f; std - %0.10f'
                        % (results_noStopWords.values(),
                        grid_go.cv_results_['mean_test_score'][i],
                        grid_go.cv_results_['std_test_score'][i]))

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(grid_go.best_estimator_, './pickled_models/review_polarity/Classifier.pkl')  

        # (3) PREDICT
        predicted = grid_go.predict(data_test)

    ## PREPARE OFF - Best Parameters are already known
    else:   

        # (1) TRAIN
        pipeline.fit(data_train, labels_train)

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/review_polarity/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

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


### LET'S BUILD : NaiveBayes

# Grid Search On
pipeline = Pipeline([
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer()),  # 1-Grams Vectorizer
                        ('vect2', CountVectorizer()),],  # 2-Grams Vectorizer
                    )),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])  

parameters = {'tfidf__use_idf': [True],
            'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
            'union__vect1__max_df': [0.90, 0.80, 0.70],
            'union__vect1__min_df': [5, 8],
            'union__vect1__ngram_range': [(1, 1)],              
            'union__vect1__stop_words': [stopwords.words("english"), 'english', stopwords_complete_lemmatized],
            'union__vect1__strip_accents': ['unicode'],
            'union__vect1__tokenizer': [LemmaTokenizer()],
            'union__vect2__max_df': [0.95, 0.85, 0.75],
            'union__vect2__min_df': [5, 8],
            'union__vect2__ngram_range': [(2, 2)],              
            'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
            'union__vect2__strip_accents': ['unicode'],
            'union__vect2__tokenizer': [LemmaTokenizer()],} 

Run_Classifier(1, 0, pipeline, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized)

# Grid Search Off

# Optimal
parameters = {'tfidf__use_idf': [True],
              'vect__max_df': [0.80],
              'vect__min_df': [5],
              'vect__ngram_range': [(1, 1)],              
              'vect__stop_words': [stopwords_complete_lemmatized],
              'vect__strip_accents': ['unicode'],
              'vect__tokenizer': [LemmaTokenizer()],}

Run_Classifier(0, 0, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized)




###


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