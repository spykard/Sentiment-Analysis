'''
Get Sentiment Weight by Word Counting in Small Phrases
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

import pandas as pd
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek 
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

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


print("Loading data...")
train = pd.read_csv("./kaggle_temp/train.tsv", sep="\t")
print("Train shape:", train.shape)

datasettargetnames = ['0', '1', '2', '3', '4']

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Split, data & labels are pairs
#data_train, data_test, labels_train, labels_test = train_test_split(train['Phrase'], train['Sentiment'], test_size=0.40, random_state=22)


### LET'S BUILD : Word Spotting and Counting using Opinion Lexicon

### Model 1
# Get Sentiment Words from a generic Opinion Lexicon
pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  

wnl = WordNetLemmatizer()
translator = str.maketrans('','', sub('\'', '', string.punctuation))

combinedEmotional = pos_words + sorted(set(neg_words) - set(pos_words))

combinedEmotionalDict = dict()
for i in range(0, len(combinedEmotional)):
    combinedEmotionalDict[combinedEmotional[i]] = [0, 0]  # Proto mideniko to Count, deutero to Sum/Count 

for index, row in train.iterrows():
    temp = []
    for t in word_tokenize(row['Phrase']):
        x = t.translate(translator) 
        if x != '': temp.append(wnl.lemmatize(x.lower())) 
    
    if (len(temp)<=4):
        for i in range(0, len(temp)):
            if (temp[i] in combinedEmotionalDict):            
                combinedEmotionalDict[temp[i]][0] += 1
                combinedEmotionalDict[temp[i]][1] += row['Sentiment']
    

todelete = []
for key, value in combinedEmotionalDict.items():
    if (value[0] == 0):
        todelete.append(key) 
    else:
        combinedEmotionalDict[key][1] = value[1] / value[0]

for k in todelete: del combinedEmotionalDict[k]

output_file = pd.DataFrame.from_dict(combinedEmotionalDict, orient='index', columns=['Count', 'Sum/Count'])
output_file.to_csv('./kaggle_temp/specialWeights.csv', index=True)


quit()