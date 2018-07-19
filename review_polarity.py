'''
Sentiment Analysis Using Different Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
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

from sklearn.datasets import load_files
import re
import os

def Run_Classifier(pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames):
    '''    Run Classifier after Preprocessing and Parameters are done    '''
    # TRAIN
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

    # PREDICT
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
    def __call__(self, doc):
        return [self.wnl.lemmatize(t.lower()) for t in word_tokenize(doc)]

# PREPROCESSING
dataset = load_files('./review_polarity/txt_sentoken', shuffle=False)

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Print Precision

# Split where X and y are pairs: data & labels
data_train, data_test, labels_train, labels_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=None)


# LET'S BUILD : NaiveBayes
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

Run_Classifier(pipeline1, parameters, data_train, data_test, labels_train, labels_test, dataset.target_names)


# LET'S BUILD : SGDC-SVM
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

#Run_Classifier(pipeline2, parameters, data_train, data_test, labels_train, labels_test)


# LET'S BUILD : SVM
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

#Run_Classifier(pipeline3, parameters, data_train, data_test, labels_train, labels_test)


# LET'S BUILD : Counting Sentimental Words

# Get Sentiment Words from a Lexicon
pos_words = []
neg_words = []
for line in open('positive-words.txt', 'r'):
    pos_words.append(wnl.lemmatize(line.rstrip()))  # Must strip Newlines

for line in open('negative-words.txt', 'r'):
    neg_words.append(wnl.lemmatize(line.rstrip()))  # Must strip Newlines

count_vect = CountVectorizer(max_df=0.80, min_df=5, analyzer='word', stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())
data_train_counts = count_vect.fit_transform(data_test)

data_array = data_train_counts.toarray()
vocabulary = count_vect.vocabulary_
final_array = np.zeros(len(data_test))  # Array of the Word Count for each Document

for word in pos_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            final_array[i] += data_array[i, vocabulary.get(word)]

for word in neg_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            final_array[i] -= data_array[i, vocabulary.get(word)]            

for i, score in enumerate(final_array):
    if score >= 0:
        final_array[i] = 1
    else:
        final_array[i] = 0

Print_Result_Metrics(labels_test, final_array, dataset.target_names)

# print('\n- - - - - RESULT METRICS - - - - -')
# print('Exact Accuracy: ', metrics.accuracy_score(labels_test, final_array))
# print(metrics.classification_report(labels_test, final_array, target_names=dataset.target_names))
# print(metrics.confusion_matrix(labels_test, final_array))


# dataActual = []
# #data_labels = []
# for i, filename in enumerate(os.listdir('./moviereviews/pos')):
#     with open('./moviereviews/pos/' + filename) as f:
#        # print(f)     
#         dataActual.append(1) 

# for filename in os.listdir('./sentoken/neg'):
#     with open('./sentoken/neg/' + filename) as f:
#         dataActual.append(1) 

# dataPredicted = []
# for i, filename in enumerate(os.listdir('./sentoken/pos')):
#     with open('./sentoken/pos/' + filename) as f:
        
#         data = f.read().replace('\n', '')
                     
#         # 2. Remove non-letters     
#         letters_only = re.sub("[^a-zA-Z]", " ", data) 
#         # 3. Convert to lower case, split into individual words
#         words = letters_only.split()

#         words = map(str.lower,words)
#         #
#         # 4. In Python, searching a set is much faster than searching
#         #   a list, so convert the stop words to a set
#         stops = set(stopwords.words("english"))                  
#         # 
#         # 5. Remove stop words
#         meaningful_words = [w for w in words if not w in stops]  

#         count = 0
#         for singleword in meaningful_words:
#             if singleword in pos_words:
#                 if i == 120:
#                     print(singleword)                
#                 count += 1
#             elif singleword in neg_words:
#                 #if i == 120:
#                 #    print(singleword)
#                 count -= 1

#         #print(str(count) + " ID " + str(i))
#         if i == 120:
#             print(data)

#         if count >= 0:
#             dataPredicted.append(1) 
#         else:
#             dataPredicted.append(0)

# for filename in os.listdir('./sentoken/neg'):
#     with open('./sentoken/neg/' + filename) as f:
#         data = f.read().replace('\n', '')
#         # 2. Remove non-letters     
#         letters_only = re.sub("[^a-zA-Z]", " ", data) 
#         # 3. Convert to lower case, split into individual words
#         words = letters_only.split()

#         words = map(str.lower,words)
#         #
#         # 4. In Python, searching a set is much faster than searching
#         #   a list, so convert the stop words to a set
#         stops = set(stopwords.words("english"))                  
#         # 
#         # 5. Remove stop words
#         meaningful_words = [w for w in words if not w in stops]  

#         count = 0
#         for singleword in meaningful_words:
#             if singleword in pos_words:
#                 count += 1
#             elif singleword in neg_words:
#                 count -= 1

#         if count >= 0:
#             dataPredicted.append(1) 
#         else:
#             dataPredicted.append(0)

# print(dataPredicted.count(0))
# print(dataPredicted.count(1))

   
