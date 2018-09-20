'''
Sentiment Analysis Using Interesting Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
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

def Run_Classifier(grid_search_enable, pickle_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, model_name):
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
            if model_name != '(MultiLayer Perceptron)':
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
        #if model_name != '(MultiLayer Perceptron)': print('\nNumber of Features/Dimension is:', pipeline.named_steps['sgdclassifier'].coef_.shape[1])

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/review_polarity/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

    Print_Result_Metrics(labels_test, predicted, targetnames, model_name)  

    return predicted

def Print_Result_Metrics(labels_test, predicted, targetnames, model_name):
    '''    Print Metrics after Training etc.    '''
    print('\n- - - - - RESULT METRICS', model_name, '- - - - -')
    print('Exact Accuracy: ', metrics.accuracy_score(labels_test, predicted))
    print(classification_report_imbalanced(labels_test, predicted, target_names=targetnames))
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

import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import CondensedNearestNeighbour # 0.6057
from imblearn.under_sampling import EditedNearestNeighbours 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

print("Loading data...")
train = pd.read_csv("./kaggle_temp/train.tsv", sep="\t")
print("Train shape:", train.shape)

datasettargetnames = ['0', '1', '2', '3', '4']

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Split, data & labels are pairs
data_train, data_test, labels_train, labels_test = train_test_split(train['Phrase'], train['Sentiment'], test_size=0.40, random_state=22)

# Dimensionality Reduction - 4 different ways to pick the best Features 
#   (1) ('feature_selection', SelectKBest(score_func=chi2, k=5000)),                    
#   (2) ('feature_selection', TruncatedSVD(n_components=1000)),  # Has Many Issues
#   (3) ('feature_selection', SelectFromModel(estimator=LinearSVC(), threshold='2.5*mean')),
#   (4) ('feature_selection', SelectFromModel(estimator=LinearSVC(penalty='l1', dual=False), threshold='mean')),  # Technically L1 is better than L2


### LET'S BUILD : SVM

# Grid Search Off
pipeline = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    #('clf', LinearSVC(loss='hinge', penalty='l2', max_iter=1000, C=500, dual=True, class_weight='balanced')),])  # dual: True for Text/High Feature Count
                    ('clf', SVC(max_iter=1000, C=500, class_weight='balanced')),])
#Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, datasettargetnames, stopwords_complete_lemmatized, '(MultiLayer Perceptron)')
###

### Logistic Regression 0.61 accuracy

# Grid Search Off
pipeline = make_pipeline_imb( # Optimal
                            FeatureUnion(transformer_list=[      
                                ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                                ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                                transformer_weights={
                                    'vect1': 1.0,
                                    'vect2': 1.0,},
                            ),
                            TfidfTransformer(use_idf=True),
                            SMOTE(ratio={0: 11500, 4: 12500}, random_state=22),
                            RandomUnderSampler(ratio={2: 19500}, random_state=22),
                            SelectKBest(score_func=chi2, k=1000),  # Dimensionality Reduction                  
                            LogisticRegression(penalty='l2', max_iter=1000, C=500, dual=True, class_weight='balanced', random_state=22),)  

predicted = Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, datasettargetnames, stopwords_complete_lemmatized, '(Logistic Regression)')
###

# SGD 0.54 accuracy

# Grid Search Off
pipeline = make_pipeline_imb( # Optimal
                            FeatureUnion(transformer_list=[      
                                ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                                ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                                transformer_weights={
                                    'vect1': 1.0,
                                    'vect2': 1.0,},
                            ),
                            TfidfTransformer(use_idf=True),
                            RandomUnderSampler(ratio={2: 24000}),
                            SelectFromModel(estimator=LinearSVC(), threshold='1.5*mean'),  # Dimensionality Reduction               
                            SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=1000, tol=None, n_jobs=-1, class_weight='balanced'),)  

#Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, datasettargetnames, stopwords_complete_lemmatized, '(SGD)')
###

# MLP 0.60

# Grid Search Off
pipeline = make_pipeline_imb( # Optimal
                            FeatureUnion(transformer_list=[      
                                ('vect1', CountVectorizer(max_df=0.80, min_df=8, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                                ('vect2', CountVectorizer(max_df=0.95, min_df=10, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                                transformer_weights={
                                    'vect1': 1.0,
                                    'vect2': 1.0,},
                            ),
                            TfidfTransformer(use_idf=True),
                            RandomUnderSampler(ratio={2: 20000}),
                            SelectFromModel(estimator=LinearSVC(), threshold='1.5*mean'),  # Dimensionality Reduction               
                            MLPClassifier(verbose=True, hidden_layer_sizes=(200,), max_iter=100, solver='sgd', learning_rate='adaptive', learning_rate_init=0.60, momentum=0.50, alpha=1e-01),)  
                            #MLPClassifier(verbose=True, random_state=22, hidden_layer_sizes=(100,), max_iter=100, solver='sgd', learning_rate='constant', learning_rate_init=0.07, momentum=0.90, alpha=1e-01),)

#predicted = Run_Classifier(0, 0, pipeline, {}, data_train, data_test, labels_train, labels_test, datasettargetnames, stopwords_complete_lemmatized, '(MultiLayer Perceptron)')
###

### Model 2
# Get Sentiment Words from a generic Opinion Lexicon

print("Loading Special Weights...")
specialweights = pd.read_csv("./kaggle_temp/specialWeights.csv", sep=",")
print("Special Weights shape:", train.shape)

specialweights = specialweights.drop('Count', 1)


pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  

wnl = WordNetLemmatizer()
translator = str.maketrans('','', sub('\'', '', string.punctuation))

for i in range(0, len(predicted)):
    # HYPOTHESIS
    if predicted == 2:
        temp = []
        for t in word_tokenize(data_test[i]):
            x = t.translate(translator) 
            if x != '': temp.append(wnl.lemmatize(x.lower())) 
        



count_vect = CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())
data_test_counts = count_vect.fit_transform(data_test)

data_array = data_test_counts.toarray()
vocabulary = count_vect.vocabulary_
final_array = np.zeros(len(data_test))  # Array of the Score for each Document
countImpact_Pos = countImpact_Neg = 0

for word in pos_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            temp = data_array[i, vocabulary.get(word)]
            final_array[i] += temp
            countImpact_Pos += np.sum(temp)

for word in neg_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            temp = data_array[i, vocabulary.get(word)]
            final_array[i] -= temp
            countImpact_Neg += np.sum(temp)   






#output_file = pd.DataFrame(data={'PhraseId':labels_test, 'Sentiment':predicted})
#output_file.to_csv('submission.csv', index=False)