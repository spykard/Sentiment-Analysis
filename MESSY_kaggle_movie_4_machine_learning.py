'''
Sentiment Analysis Using Interesting Techniques. Kaggle Dataset of Rotten Tomatoes Movie Review
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

# kaggle_4
import gzip

def load(file_name):
    # load the model
    with open(file_name, "rb") as handle:
        x = pickle.load(handle)
    return x

def save(file_name, model):
    # save the model
    with open(file_name, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def Run_Classifier(grid_search_enable, pickle_enable, unlabeled_test_enable, pipeline, parameters, data_train, data_test, labels_train, labels_test, targetnames, stopwords_complete_lemmatized, model_name):
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
        if model_name != 'mlpclassifier': print('\nNumber of Features/Dimension is:', pipeline.named_steps[model_name].coef_.shape[1])

        # (2) Model Persistence (Pickle)
        if pickle_enable == 1: joblib.dump(pipeline, './pickled_models/kaggle_review/Classifier.pkl') 

        # (3) PREDICT
        predicted = pipeline.predict(data_test)

    if unlabeled_test_enable == 0:
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
from imblearn.under_sampling import ClusterCentroids 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.base import TransformerMixin, BaseEstimator
import pickle

print("Loading data...")
train = pd.read_csv("./datasets/kaggle_movie/train.tsv", sep="\t")
print("Train shape:", train.shape)

testset = pd.read_csv("./datasets/kaggle_movie/test.tsv", sep="\t")
print("Train shape:", testset.shape)


datasettargetnames = ['0', '1', '2', '3', '4']

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# # Remove Instances from the Data that will have a vector of only Zeros
# count_vect_1 = FeatureUnion(transformer_list=[      
#                         ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
#                         ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

#                         transformer_weights={
#                             'vect1': 1.0,
#                             'vect2': 1.0,},
#                     )

# data_test_counts_1 = count_vect_1.fit_transform(train['Phrase'])

# total_x, total_y = data_test_counts_1.shape

# #print(data_test_counts_1[3])
# #print(data_test_counts_1[0, 2345])

# #data_array_1 = data_test_counts_1.toarray()
# #todelete = np.where(data_test_counts_1 == 0)
# x_nonzero, y_nonzero = np.nonzero(data_test_counts_1)

# #print(total_x, total_y)
# #print(len(set(x_nonzero)), len(set(y_nonzero)))
# x_zero = list(set(range(0, total_x-1)) - set(x_nonzero))
# y_zero = list(set(range(0, total_y-1)) - set(y_nonzero))

# print("Removing", len(x_zero), "empty Instances from Data...")

# train = train.drop(train.index[x_zero])


# print(load("./pickled_models/kaggle_movie/PredictedRegression.pkl").shape)
# quit()

# train.hist(column="Sentiment")
# plt.xlabel("Trip_distance",fontsize=15)
# plt.ylabel("Frequency",fontsize=15)
# plt.show()
# quit()

# Split, data & labels are pairs
data_train, data_test, labels_train, labels_test = train_test_split(train['Phrase'], train['Sentiment'], test_size=0.20, random_state=22)

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

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
                            #RandomUnderSampler(ratio={2: 32000}, random_state=22),
                            SelectFromModel(estimator=LinearSVC(), threshold='1.2*mean'),  # Dimensionality Reduction               
                            #MLPClassifier(verbose=True, hidden_layer_sizes=(200,), max_iter=100, solver='sgd', learning_rate='adaptive', learning_rate_init=0.60, momentum=0.50, alpha=1e-01),)  
                            MLPClassifier(verbose=True, random_state=22, hidden_layer_sizes=(100,), max_iter=100, solver='sgd', learning_rate='constant', learning_rate_init=0.07, momentum=0.90, alpha=1e-01),)

#predicted = Run_Classifier(0, 0, 1, pipeline, {}, train['Phrase'], testset['Phrase'], train['Sentiment'], [], datasettargetnames, stopwords_complete_lemmatized, 'mlpclassifier')
###

#save("./pickled_models/kaggle_movie/PredictedNeuralNetwork.pkl", predicted)
#quit()

### Model 2

### Logistic Regression 0.61 accuracy

# Grid Search Off
pipeline = make_pipeline_imb( # Optimal
                            FeatureUnion(transformer_list=[      
                                ('vect1', TfidfVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer(), sublinear_tf=True, analyzer='word')),  # 1-Gram Vectorizer
                                ('vect2', TfidfVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 3), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer(), sublinear_tf=True, analyzer='word')),],  # 2-Gram Vectorizer

                                transformer_weights={
                                    'vect1': 1.0,
                                    'vect2': 1.0,},
                            ),
                            TfidfTransformer(use_idf=True),
                            #SMOTE(ratio={0: 12500, 4: 14500}, random_state=22),
                            #RandomUnderSampler(ratio={1: 19000, 2: 27200, 3: 20000}, random_state=22),
                            # Note  k = 5000, max_iter 1000
                            SelectKBest(score_func=chi2, k=18000),  # Dimensionality Reduction                  
                            LogisticRegression(solver='sag', penalty='l2', max_iter=1000, C=500, dual=True, class_weight='balanced', random_state=22),)  

#predicted = Run_Classifier(0, 0, 1, pipeline, {}, train['Phrase'], testset['Phrase'], train['Sentiment'], [], datasettargetnames, stopwords_complete_lemmatized, 'logisticregression')
###

# trainplustest = pd.concat([train['Phrase'], testset['Phrase']])

# vectorizer1 = FeatureUnion(transformer_list=[      
#                                 ('vect1', TfidfVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer(), sublinear_tf=True, analyzer='word')),  # 1-Gram Vectorizer
#                                 ('vect2', TfidfVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 3), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer(), sublinear_tf=True, analyzer='word')),],  # 2-Gram Vectorizer

#                                 transformer_weights={
#                                     'vect1': 1.0,
#                                     'vect2': 1.0,},
#                             )

# vectorizer1.fit(trainplustest)
# train_word_features = vectorizer1.transform(train['Phrase'])
# test_word_features = vectorizer1.transform(testset['Phrase'])



#save("./pickled_models/kaggle_movie/PredictedRegression.pkl", predicted)
#quit()

# Get Sentiment Words from a generic Opinion Lexicon

print("Loading Special Weights...")
specialweights = pd.read_csv("./opinion_lexicon/kaggle_movie_sentiment-words-weights.csv", sep=",")
print("Special Weights shape:", specialweights.shape)

specialweights.columns = ['Word', 'Count', 'Sum/Count']
specialweights.set_index(['Word'])

specialweights = specialweights.drop('Count', 1)

specialWords = specialweights['Word'].tolist()
specialScore = specialweights['Sum/Count'].tolist()

pos_words = []
neg_words = []
for line in open('./opinion_lexicon/positive-words.txt', 'r'):
    pos_words.append(line.rstrip())  # Must strip Newlines

for line in open('./opinion_lexicon/negative-words.txt', 'r'):
    neg_words.append(line.rstrip())  # Must strip Newlines  

#print(load("./pickled_models/kaggle_movie/CountTest.pkl")[0:50])
#quit()


# count_vect = CountVectorizer(max_df=0.95, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())
# data_test_counts = count_vect.fit_transform(testset['Phrase'])

# data_array = data_test_counts.toarray()
# vocabulary = count_vect.vocabulary_
# final_array = np.zeros((len(testset['Phrase']), 3))  # Array of the Count/Sum on 0, Count on 1 and Sum on 2
# countImpact_Pos = countImpact_Neg = 0

# combinePosNeg = pos_words + sorted(set(neg_words) - set(pos_words))

# #final_array[1][0] = 2
# #save("./pickled_models/kaggle_movie/Count.pkl", final_array[:][0])
# #print(load("/.pickled_models/kaggle_movie/Count.pkl"))

# minimizedNewDict = {}

# for word in combinePosNeg:
#     if word in vocabulary:
#         minimizedNewDict[word]=vocabulary[word]

# count = 0

# for word in minimizedNewDict:  # For each Sentimental Word update the Array
#     if word in specialWords:
#         sScore = specialScore[specialWords.index(word)]
#     for i in range(0, len(data_test)):
#         temp = data_array[i, minimizedNewDict.get(word)]
#         if temp > 0:
#             if word in specialWords:
#                 # print ('word: ', word, '\t i: ', i, '\t sScore: ', sScore)
#                 final_array[i][1] += temp
#                 final_array[i][2] += sScore * temp # I have a specific Label recomendation
#                 #print(i)
#                 #print(final_array[i][2])
#                 #print(final_array[i+1][2])
#             elif word in pos_words:
#                 final_array[i][1] += temp
#                 final_array[i][2] += 3 * temp # I have a specific Label recomendation
#                 countImpact_Pos += 1
#             elif word in neg_words:
#                 final_array[i][1] += temp
#                 final_array[i][2] += 1 * temp # I have a specific Label recomendation
#                 countImpact_Neg += 1    
#     count += 1
#     print(count)
#     # if count == 100: 
#     #     break

# for y in range(0, len(data_test)):
#     if final_array[y][1] != 0:
#         final_array[y][0] = (final_array[y][2]) / (final_array[y][1])  

# print(final_array[0:50])

# save("./pickled_models/kaggle_movie/CountTest.pkl", final_array)
# quit()

final_array_final = load("./pickled_models/kaggle_movie/CountTest.pkl")

# x = final_array_final[np.argsort(final_array_final[:, 1])]
# print(x[50000:52000])
# quit()

final_array_countstrength = final_array_final[:,1]
final_array_final = final_array_final[:,0]

# for i, score in enumerate(final_array_final):
#     # rounding = int(round(score))
#     if score = 0:  # Osa den eixan kamia leksi mesa einai mazi me ta negatives distixos
#         print('FUCK')
#     if score < 0.7:
#         final_array_final[i] = 0
#     elif score < 1.69:
#         final_array_final[i] = 1
#     elif score < 2.6:
#         final_array_final[i] = 2
#     elif score < 3.3:
#         final_array_final[i] = 3
#     elif score < 4.0:
#         final_array_final[i] = 4                       
        
#Print_Result_Metrics(labels_test, final_array_final, datasettargetnames, '')  


predicted = load("./pickled_models/kaggle_movie/PredictedRegression.pkl")

#Print_Result_Metrics(labels_test, predicted, datasettargetnames, '') 

for i, score in enumerate(final_array_final):
    # rounding = int(round(score))
    # IF WE HAVE A RECOMMENDATION
    if final_array_countstrength[i] > 0:
        if score < 0.72:
            rounding = 0
        elif score < 1.75:
            rounding = 1
        elif score < 2.6:
            rounding = 2
        elif score < 3.36:
            rounding = 3
        elif score < 4.0:
            rounding = 4  

        diff = abs(rounding - predicted[i])
        if (predicted[i] - rounding) > 0:
            side = 'L'
        elif (predicted[i] - rounding) > 0:
            side = 'R'
        else:
            side = 'EQ'

        if diff >= 2:         
            if predicted[i] == 0:       ###
                if diff >= 3:  # Megali Diafora, parto poli deksia
                    predicted[i] = 2  
                else:         # Mikri Diafora, parto mono mia thesi deksia
                    if final_array_countstrength[i] > 0: # megalo STRENGTH/CONFIDENCE
                        predicted[i] = 1
            elif predicted[i] == 1:     ###
                if diff >= 3:  # Megali Diafora, kounato poli                    
                    predicted[i] = 3  
                else:         # Mikri Diafora, kounato mono mia thesi
                    if side == 'L':
                        predicted[i] = 0  
                    elif side == 'R':
                        predicted[i] = 2 
            elif predicted[i] == 2:     ###
                # NO MATTER WHAT MONO MIA THESI
                predicted[i] = rounding
                # if side == 'L':
                #     predicted[i] = 1  
                # elif side == 'R':
                #     predicted[i] = 3 
            elif predicted[i] == 3:     ###
                if diff >= 3:  # Megali Diafora, kounato poli                    
                    predicted[i] = 1  
                else:         # Mikri Diafora, kounato mono mia thesi
                    if side == 'L':
                        predicted[i] = 2  
                    elif side == 'R':
                        predicted[i] = 4 
            elif predicted[i] == 4:     ###
                if diff >= 3:  # Megali Diafora, parto poli deksia
                    predicted[i] = 2  
                else:         # Mikri Diafora, parto mono mia thesi deksia
                    if final_array_countstrength[i] > 0: # megalo STRENGTH/CONFIDENCE
                        predicted[i] = 3

#Print_Result_Metrics(labels_test, predicted, datasettargetnames, '') 

#save("./pickled_models/kaggle_movie/PredictedRegressionPlusCounting.pkl", predicted)
#quit()

### Model 3 COMBINE ALL

neuralnetwork = load("./pickled_models/kaggle_movie/PredictedNeuralNetwork.pkl")
logisticregression = load("./pickled_models/kaggle_movie/PredictedRegressionPlusCounting.pkl")

#Print_Result_Metrics(labels_test, neuralnetwork, datasettargetnames, '')

for i, score in enumerate(logisticregression):
    if (score == 0) or (score == 4):
        neuralnetwork[i] = score

save("./pickled_models/kaggle_movie/FinalCombined.pkl", neuralnetwork)
quit()

#Print_Result_Metrics(labels_test, neuralnetwork, datasettargetnames, '') 

# output_file = pd.DataFrame(data={'PhraseId':labels_test, 'Sentiment':predicted})
# output_file.to_csv('submission.csv', index=False)