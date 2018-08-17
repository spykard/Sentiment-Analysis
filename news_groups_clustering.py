'''
Birch Spherical K-Means Clustering (Unsupervised Machine Learning). 20 Newsgroups Dataset
'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
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
import random

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
dataset = load_files('./datasets/news_groups', encoding='latin1', shuffle=False)  # Load all Categories

stopwords_complete = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
wnl = WordNetLemmatizer()
stopwords_complete_lemmatized = set([wnl.lemmatize(word) for word in stopwords_complete])

np.set_printoptions(precision=10)  # Numpy Precision when Printing

# Select only Part of the Dataset/Instances
data, data_unused, labels, labels_unused = train_test_split(dataset.data, dataset.target, stratify=dataset.target, test_size=0.50)

print('\nDocuments: ', len(labels))
print('Categories: ', len(dataset.target_names))


### LET'S BUILD : Birch Spherical K-Means Clustering

pipeline1 = Pipeline([ # Optimal
                    ('union', FeatureUnion(transformer_list=[      
                        ('vect1', CountVectorizer(max_df=0.80, min_df=5, ngram_range=(1, 1), stop_words=stopwords_complete_lemmatized, strip_accents='unicode', tokenizer=LemmaTokenizer())),  # 1-Gram Vectorizer
                        ('vect2', CountVectorizer(max_df=0.95, min_df=8, ngram_range=(2, 2), stop_words=None, strip_accents='unicode', tokenizer=LemmaTokenizer())),],  # 2-Gram Vectorizer

                        transformer_weights={
                            'vect1': 1.0,
                            'vect2': 1.0,},
                    )),
                    ('tfidf', TfidfTransformer(use_idf=True)),])

X = pipeline1.fit_transform(data)

# Note :
# Vectorizer results are normalized, which makes KMeans behave as spherical k-means for better results. 
# Since LSA/SVD results are not normalized, we have to redo the normalization.
pipeline2 = Pipeline([
                    ('svd', TruncatedSVD(800)),  
                    ('norm', Normalizer(copy=False)),])

X = pipeline2.fit_transform(X)

print('\nNumber of Features/Dimension is:', X.shape[1], '\n') 


### Birch
# Trial and Error, until we get the Root to have as many Children as there are Categories
true_k = len(dataset.target_names)
while True: 
    tresholdSplit = random.uniform(0.25, 0.60) 
    birch = Birch(n_clusters=20, threshold=tresholdSplit)  
    print('Birch Clustering Attempt with threshold:', tresholdSplit)
    birch.fit(X)

    print(len(birch.root_.subclusters_), 'vs.', true_k)
    # print(len(birch.subcluster_centers_))
    if len(birch.root_.subclusters_) == true_k: break

print("\n(Birch Classic)")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, birch.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, birch.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, birch.labels_))
print("Adjusted Rand-Index: %0.3f"
      % metrics.adjusted_rand_score(labels, birch.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, birch.labels_, sample_size=1000))
###


input_to_KMeans = list()
for cluster_from_root in birch.root_.subclusters_:
    # print(cluster_from_root.n_samples_)
    # print(cluster_from_root.centroid_)    
    input_to_KMeans.append(list(cluster_from_root.centroid_))

input_to_KMeans = np.asarray(input_to_KMeans)
for i, j in enumerate(input_to_KMeans):
    input_to_KMeans[i] = np.array(j)


# K-Means Classic
km1 = KMeans(n_clusters=true_k, init='random', max_iter=300, n_init=1, verbose=False) 
km1.fit(X)

print("\n(K-Means Classic)")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km1.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km1.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km1.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km1.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km1.labels_, sample_size=1000))  

# K-Means with SciKit's Technique
km2 = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1, verbose=False)
km2.fit(X)

print("\n(K-Means with SciKit's Technique)")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km2.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km2.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km2.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km2.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km2.labels_, sample_size=1000))  

#  K-Means with Birch
km3 = KMeans(n_clusters=true_k, init=input_to_KMeans, max_iter=300, n_init=1, verbose=False)               
km3.fit(X)

print("\n([!] K-Means with Birch)")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km3.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km3.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km3.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km3.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km3.labels_, sample_size=1000))


### Print Top Terms per Cluster
original_space_centroids = pipeline2.named_steps['svd'].inverse_transform(km3.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

features = [w[7:] for w in pipeline1.named_steps['union'].get_feature_names()]  # Get the Features from a Pipeline+Union
print('\n\nTop terms per cluster:')
for i in range(true_k):
    print('Cluster %d:' % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s,' % features[ind], end='')
    print()    
###