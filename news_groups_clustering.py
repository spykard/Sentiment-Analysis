from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans, Birch

import logging
import random
from optparse import OptionParser
import sys
from time import time

import numpy as np

###
# Load only some Categories
#categories = [
#    'alt.atheism',
#    'talk.religion.misc',
#    'comp.graphics',
#    'sci.space',
#]
###

print()
print("Loading 20 newsgroups dataset for categories:")

from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_files

# dataset = fetch_20newsgroups(subset='all', shuffle=False, random_state=22)
dataset = load_files('./datasets/news_groups', encoding='latin1', shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, stratify=dataset.target, test_size=0.95)

print("%d documents" % len(y_train))
print("%d categories" % len(dataset.target_names))
print()

labels = y_train
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a vectorizer")

vectorizer = TfidfVectorizer(max_df=0.5, max_features=100,
                                min_df=2, stop_words='english',
                                use_idf=True)
X = vectorizer.fit_transform(X_train)

print("n_samples: %d, n_features: %d" % X.shape)
print()


print("Performing dimensionality reduction using LSA")
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(99)
normalizer = Normalizer(copy=False) # IMPORTANT
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

print()

print(X.shape)

# #############################################################################
# Do the actual clustering
while True:  # Trial and Error until we get the Root to have 4 Children
    if true_k == 4:
        # tresholdSplit = random.uniform(0.86, 0.92)
        tresholdSplit = random.uniform(0.945, 0.971)
    elif true_k == 20:
        # tresholdSplit = random.uniform(0.405, 0.44)
        tresholdSplit = random.uniform(0.40, 0.70)

    birch = Birch(n_clusters=20, threshold=tresholdSplit)  
    print("Clustering sparse data with %s" % birch)
    birch.fit(X) 

    # print(len(birch.subcluster_centers_))
    print(true_k, len(birch.root_.subclusters_))

    if len(birch.root_.subclusters_) == true_k: break

inputToKMeansArray = list()

for clusterFromRoot in birch.root_.subclusters_:
    # print(clusterFromRoot.n_samples_)
    # print(clusterFromRoot.centroid_)    
    inputToKMeansArray.append(list(clusterFromRoot.centroid_))

np.array(inputToKMeansArray)
for i, j in enumerate(inputToKMeansArray):
    inputToKMeansArray[i] = np.array(j)

# With Birch
km = KMeans(n_clusters=true_k, init=np.array(inputToKMeansArray), max_iter=300, n_init=1,
                  verbose=False)               

# Without Birch
km2 = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                  verbose=False)     

print("Clustering sparse data with %s" % km)

print("\n(Birch)")
km.fit(X)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, birch.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, birch.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, birch.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, birch.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, birch.labels_, sample_size=1000))

print("\n(Birch then KMeans)")
km.fit(X)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print("\n(KMeans)")
km2.fit(X)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km2.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km2.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km2.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km2.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km2.labels_, sample_size=1000))  

print()

# if not opts.use_hashing:
#     print("Top terms per cluster:")

#     if opts.n_components:
#         original_space_centroids = svd.inverse_transform(km.cluster_centers_)
#         order_centroids = original_space_centroids.argsort()[:, ::-1]
#     else:
#         order_centroids = km.cluster_centers_.argsort()[:, ::-1]

#     terms = vectorizer.get_feature_names()
#     for i in range(true_k):
#         print("Cluster %d:" % i, end='')
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind], end='')
#         print()
