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



def logistic_regression_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_logistic('data/Google_data.csv', n_groups=8)
    model1 = LogisticRegression()
    model2 = LogisticRegression()
    model1.fit(X_no_text_train,y_no_text_train)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    model2.fit(X_with_text_train,y_with_text_train)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)
    print('Accuracy Score of the model without text: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text: {}'.format(text_model_score))


if __name__ == '__main__':
    data_path = 'data/Google_data.csv'
    logistic_regression_model(data_path)
