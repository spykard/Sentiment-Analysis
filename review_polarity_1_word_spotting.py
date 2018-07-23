'''
Sentiment Analysis Using Interesting Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
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

def Run_Classifier(data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, pickle_enable):
    '''    Run Classifier to be used as an Opinion Pos/Neg Lexicon (List of Positive and Negative Words)    '''
    ## PREPARE
    # Grid Search used to Look for the Best Parameters

    # pipeline = Pipeline([
    #                     ('union', FeatureUnion(transformer_list=[      
    #                         ('vect1', CountVectorizer()),  # 1-Gram Vectorizer
    #                         ('vect2', CountVectorizer()),],  # 2-Gram Vectorizer
    #                     )),

    #                     ('tfidf', TfidfTransformer()),
    #                     ('clf', MultinomialNB()),])       
    # parameters = {'tfidf__use_idf': [True],
    #             'union__transformer_weights': [{'vect1':1.0, 'vect2':1.0},],
    #             'union__vect1__max_df': [0.90, 0.80, 0.70],
    #             'union__vect1__min_df': [5, 8],
    #             'union__vect1__ngram_range': [(1, 1)],              
    #             'union__vect1__stop_words': [stopwords_complete_lemmatized],
    #             'union__vect1__strip_accents': ['unicode'],
    #             'union__vect1__tokenizer': [LemmaTokenizer()],
    #             'union__vect2__max_df': [0.95, 0.85, 0.75],
    #             'union__vect2__min_df': [5, 8],
    #             'union__vect2__ngram_range': [(2, 2)],              
    #             'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
    #             'union__vect2__strip_accents': ['unicode'],
    #             'union__vect2__tokenizer': [LemmaTokenizer()],} 

    ## TRAIN
    # grid_go = GridSearchCV(pipeline, parameters, n_jobs=-1)
    # grid_go = grid_go.fit(data_train, labels_train)
    # print('- - - - - BEST PARAMETERS - - - - -')
    # print(grid_go.best_score_, 'Accuracy')
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, grid_go.best_params_[param_name]))

    # print('\n- - - - - DETAILS - - - - -')
    # for i in range(len(grid_go.cv_results_['params'])):
    #     results_noStopWords = copy.deepcopy(grid_go.cv_results_['params'][i])
    #     if results_noStopWords['union__vect1__stop_words'] is not None:  # Don't Print the list of Stopwords
    #         results_noStopWords['union__vect1__stop_words'] = ['ListOfStopWords']   
    #     if results_noStopWords['union__vect2__stop_words'] is not None:
    #         results_noStopWords['union__vect2__stop_words'] = ['ListOfStopWords']           
    #     print(i, 'params - %s; mean - %0.10f; std - %0.10f'
    #                 % (results_noStopWords.values(),
    #                 grid_go.cv_results_['mean_test_score'][i],
    #                 grid_go.cv_results_['std_test_score'][i]))

    # Optimal
    pipeline = Pipeline([
                        ('union', FeatureUnion(transformer_list=[      
                            ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                            ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                            transformer_weights={
                                'vect1': 1.0,
                                'vect2': 1.0,},
                        )),

                        ('tfidf', TfidfTransformer(use_idf=True)),
                        ('clf', MultinomialNB()),])     

    ## TRAIN
    pipeline.fit(data_train, labels_train)

    # Model Persistence (Pickle)
    if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/review_polarity/TrainedBagOfWords.pkl') 

    ## PREDICT
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

# Build
Run_Classifier(data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized, 1)
# or Load
#clf = joblib.load('./pickled_models/review_polarity/TrainedBagOfWords.pkl')


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


### LET'S BUILD : Word Spotting and Counting using Opinion Lexicon

# Model 1
# Get Sentiment Words from an Opinion Lexicon
pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  

count_vect = CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())
data_test_counts = count_vect.fit_transform(data_test)

data_array = data_test_counts.toarray()
vocabulary = count_vect.vocabulary_
final_array = np.zeros(len(data_test))  # Array of the Score for each Document

for word in pos_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            final_array[i] += data_array[i, vocabulary.get(word)]

for word in neg_words:  # For each Sentimental Word update the Array
    if word in vocabulary:
        for i in range(0, len(data_test)):
            final_array[i] -= data_array[i, vocabulary.get(word)]            

## Also Count the Accuracy of every Score individually
accuracy_per_single_score = np.zeros(int(max(final_array) - min(final_array) + 1.0))  # Number of Individual Scores
accuracy_per_single_score_labels = []

addIndex = int(abs(min(final_array)))  # Add to make all indexes of Array >= 0
for j in range(int(min(final_array)), int(max(final_array)) + 1):
#for j in range(-20, 20):
    countCorrect = 0
    countTotal = 0
    for i, score in enumerate(final_array):
        if (int(score) == j):
            if (score >= 0 and labels_test[i] == 1):
                countCorrect += 1
            elif (score < 0 and labels_test[i] == 0):
                countCorrect += 1
            countTotal += 1    

    if j % 5 == 0: accuracy_per_single_score_labels.append(j)  # j mod 5     
    if countTotal != 0: accuracy_per_single_score[j + addIndex] = float(countCorrect) / countTotal

#Plot
x = np.arange(int(min(final_array)), int(max(final_array)) + 1)

fig, ax = plt.subplots()
plt.xlabel('Sentiment Score\n\u2190 Strongly Negative | Strongly Positive\u2192')
plt.ylabel('Accuracy (%)')
plt.title('Emotional Keyword Counting/Spotting Classifier\nAccuracy per individual Score')
plt.bar(x, accuracy_per_single_score)
#ax.set_xlim([-50,50])
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set

#plt.show()
## ^ SHOW ^

for i, score in enumerate(final_array):
    if score >= 1:
        final_array[i] = 1
    else:
        final_array[i] = 0

Print_Result_Metrics(labels_test, final_array, dataset.target_names)  