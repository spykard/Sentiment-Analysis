'''
Sentiment Analysis Using Interesting Techniques. Bo Pang and Lillian Lee (ACL 2004) Dataset of Movie Reviews
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
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

def Run_Classifier(grid_search_enable, pickle_enable, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized):
    '''    Run Classifier to be used as an Opinion Pos/Neg Lexicon (List of Positive and Negative Words)    '''
    
    ## PREPARE ON - Grid Search to Look for the Best Parameters
    if grid_search_enable == 1:
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
                    'union__vect1__stop_words': [stopwords_complete_lemmatized],
                    'union__vect1__strip_accents': ['unicode'],
                    'union__vect1__tokenizer': [LemmaTokenizer()],
                    'union__vect2__max_df': [0.95, 0.85, 0.75],
                    'union__vect2__min_df': [5, 8],
                    'union__vect2__ngram_range': [(2, 2)],              
                    'union__vect2__stop_words': [stopwords_complete_lemmatized, None],
                    'union__vect2__strip_accents': ['unicode'],
                    'union__vect2__tokenizer': [LemmaTokenizer()],} 

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
        if pickle_enable == 1: joblib.dump(grid_go.best_estimator_, './pickled_models/review_polarity/TrainedBagOfWords.pkl')  

        # (3) PREDICT
        predicted = grid_go.predict(data_test)
    
    ## PREPARE OFF - Best Parameters are already known
    else:   
        pipeline = Pipeline([ # Optimal
                            ('union', FeatureUnion(transformer_list=[      
                                ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                                ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                                transformer_weights={
                                    'vect1': 1.0,
                                    'vect2': 1.0,},
                            )),

                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('clf', MultinomialNB()),])     

        # (1) TRAIN
        pipeline.fit(data_train, labels_train)

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/review_polarity/TrainedBagOfWords.pkl') 

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

# Split, data & labels are pairs
data_train, data_test, labels_train, labels_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=22)

# Create
#Run_Classifier(0, 1, data_train, data_test, labels_train, labels_test, dataset.target_names, stopwords_complete_lemmatized)
# or Load
clf = joblib.load('./pickled_models/review_polarity/TrainedBagOfWords.pkl')


### LET'S BUILD : Word Spotting and Counting using SentiWordNet and Word-Sense Disambiguation

### Model 1
# Get Sentiment Words from a generic Opinion Lexicon and/or from our pickled Pos/Neg Classifier (Opinion Lexicon)
# For each sentence run a word disambiguation in WordNet then get the sentiment of it's meaning through SentiWordNet

def NLTK_to_Wordnet_PoS(tag):
    #n - Noun
    #v - Verb
    #a - Adjective
    #s - Adjective Satellite
    #r - Adverb 
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    else:
        return None  # None: will return all Synsets | '': will return no Synsets

def Find_WordNet_Path_Disamb(word, lemma, lemma_pos, tokenized_sentence, pos_tags_enable):
    '''    Disambiguate the meaning of a Word by checking Path Similarities    '''
    while word in tokenized_sentence: tokenized_sentence.remove(word)  # Remove the Word itself from Comparisons 

    # Using PoS Tags is much Faster but reduces Accuracy
    if pos_tags_enable == 1:
        word_synsets = list(wn.synsets(lemma, NLTK_to_Wordnet_PoS(lemma_pos)))
    else:
        word_synsets = list(wn.synsets(lemma))

    maxscore = 0.0
    maxlabel = None
    sentence_pos = pos_tag(tokenized_sentence)

    for synset in word_synsets:
        # Debug
        # print('\nExamining the synset: ', synset)    
        for i, word_t in enumerate(tokenized_sentence):
            temp = 0.0
            if word_t in string.punctuation: continue  # Quick Fix because of the Fast Preprocessing that didn't remove Punctuation
            if pos_tags_enable == 1: 
                temp_synsets = list(wn.synsets(word_t, NLTK_to_Wordnet_PoS(sentence_pos[i][1])))
            else:
                temp_synsets = list(wn.synsets(word_t))

            for synset2 in temp_synsets:
                similarity = synset.path_similarity(synset2)
                if (similarity is not None):
                    # Debug
                    # print('Current word is: ', i, '  Current synset2 is: ', synset2, 'with similarity', similarity)
                    if (similarity > temp):
                        temp = similarity
            if temp > maxscore:
                maxscore = temp
                maxlabel = synset
    return maxlabel

# Get Sentiment Words from a generic Opinion Lexicon
pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  


final_array = np.zeros(len(data_test))  # Array of the Score for each Document
countImpact_Pos = countImpact_Neg = 0
translator = str.maketrans('','', sub('\'', '', string.punctuation))

for i, doc in enumerate(data_test):
    ## Preprocessing
    # Convert form Byte to UTF, remove Newlines and Split Sentences
    scorePos = scoreNeg = 0.0
    split_sentences = sent_tokenize(str(doc, 'utf-8').replace('\n', ''))

    for sentence in split_sentences:
        # Correct Preprocessing but much Slower
        # tokenized_sentence = []
        # for t in word_tokenize(sentence):
        #    x = t.translate(translator) 
        #    if x != '': tokenized_sentence.append(wnl.lemmatize(x.lower())) 
        tokenized_sentence = word_tokenize(sentence)

        for j, word in enumerate(tokenized_sentence):
            # Faster
            x = word.translate(translator) 
            if x != '': 
                lemmatized = wnl.lemmatize(x.lower())
            else:
                continue
            if lemmatized in stopwords_complete_lemmatized: continue
    ##

            if ((lemmatized in pos_words) or (lemmatized in neg_words)):  # Faster at the cost of 0.02 Accuracy
                disambiguate = Find_WordNet_Path_Disamb(word, lemmatized, pos_tag(tokenized_sentence)[j][1], tokenized_sentence, 0)
                if disambiguate is not None:
                    if clf.predict([word]) == [1]:  # Positive
                        scorePos += swn.senti_synset(disambiguate.name()).pos_score()
                    else:                           # Negative
                        scoreNeg += swn.senti_synset(disambiguate.name()).neg_score()
                    # Alternative Way    
                    #if lemmatized in pos_words:
                    #if lemmatized in neg_words:

    countImpact_Pos += scorePos
    countImpact_Neg += scoreNeg
    if (scorePos - scoreNeg >= -0.9):  # Default: 0
        final_array[i] = 1
    else:
        final_array[i] = 0
    # Debug
    # print(final_array[i], labels_test[i], ' | Score: ', scorePos - scoreNeg)

print('\n- [Model 1] Impact of Positive Words:', countImpact_Pos, '| Impact of Negative Words:', countImpact_Neg, ' // Skew the Decision Boundary (Default: 0) according to the Difference')  # A word is considered Positive if it's score was bigger than 0. Depending on the Impact Difference other numbers are chosen instead of 0
###


### Model 2 on top of Model 1
# Get Sentiment Words from our pickled Pos/Neg Opinion Lexicon (Classifier)
# Flip the Polarity of confident (extreme) cases
ids_to_flip_to_Pos = []
ids_to_flip_to_Neg = []
model2_array = np.zeros(len(data_test))  # Array of the Word Count for each Document
countImpact_Pos = countImpact_Neg = 0

# Get the Features from a Pipeline+Union
# trained_features = [w[7:] for w in clf.named_steps['union'].get_feature_names()]

count_vect = CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())
data_test_counts = count_vect.fit_transform(data_test)

    # Old Version not Using Sparse, New Version at kaggle_4 
data_array = data_test_counts.toarray()
vocabulary = count_vect.vocabulary_

for word in vocabulary:
    if clf.predict([word]) == [1]:
        for i in range(0, len(data_test)):  # Positive
            temp = data_array[i, vocabulary.get(word)]
            model2_array[i] += temp
            countImpact_Pos += np.sum(temp)
    else:
        for i in range(0, len(data_test)):  # Negative
            temp = data_array[i, vocabulary.get(word)]
            model2_array[i] -= temp
            countImpact_Neg += np.sum(temp)
 
# Very high Scores that have to be Flipped to Positives/Negatives
for i, score in enumerate(model2_array):
    if score >= -25:
        ids_to_flip_to_Pos.append(i)
    elif score <= -65:
        ids_to_flip_to_Neg.append(i)

for i in ids_to_flip_to_Pos:
    if final_array[i] == 0:
        final_array[i] = 1
for i in ids_to_flip_to_Neg:
    if final_array[i] == 1:
        final_array[i] = 0

print('\n- [Model 2] Impact of Positive Words:', countImpact_Pos, '| Impact of Negative Words:', countImpact_Neg, ' //')  # A word is considered Positive if it's score was bigger than 0. Depending on the Impact Difference other numbers are chosen instead of 0
###

Print_Result_Metrics(labels_test, final_array, dataset.target_names)  