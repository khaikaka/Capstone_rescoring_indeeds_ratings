from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import random
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from data_cleaning_before_modeling import *
from sklearn.model_selection import train_test_split
from extra_cleaning_obj1 import *
import itertools


def prepare_data_for_KMEANS(data_path):
    data = pulling_data(data_path)
    data['text_reviews'][data['text_reviews'].isnull()] = ''
    data['former_current'][data['former_current'].isnull()] = 1

    data['all_text'] = data['review_titles'] + data['text_reviews']
    data['all_text'][data['all_text'].isnull()] = ''
    data = data.drop(columns=['user_ids', 'review_titles', 'text_reviews', 'position', 'city'], axis=1)

    Text = data['all_text']
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, max_features= 1000)
    vector = vectorizer.fit_transform(Text).todense()

    return vector


def grid_search_KMEANS(data_path, max_n_grp = 20):

    vector = prepare_data_for_KMEANS(data_path)

    maxk = max_n_grp
    wcss = np.zeros(maxk)
    silhouette = np.zeros(maxk)


# flatten

    for k in range(1,maxk):
        km = KMeans(k)
        y = km.fit_predict(vector)


    for c in range(0, k):
        for i1, i2 in itertools.combinations([ i for i in range(len(y)) if y[i] == c ], 2):
            wcss[k] += sum(vector[i1] - vector[i2])**2
    wcss[k] /= 2

    fig, ax = plt.subplots()
    ax.plot(range(3,maxk), wcss[3:maxk], 'o-')
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("within-cluster sum of squares")
