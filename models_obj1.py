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
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss


def logistic_regression_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_logistic(data_path, n_groups=6)
    model1 = LogisticRegression()
    model2 = LogisticRegression()
    model1.fit(X_no_text_train,y_no_text_train)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    model2.fit(X_with_text_train,y_with_text_train)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)
    print('Accuracy Score of the model without text_logistic regression: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_logistic regression: {}'.format(text_model_score))

def ridge_regression_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path, n_groups=6)
    model1 = RidgeClassifier(alpha=0.1)
    model2 = RidgeClassifier(alpha=0.1)
    model1.fit(X_no_text_train,y_no_text_train)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    model2.fit(X_with_text_train,y_with_text_train)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)
    y_pred_no_text = model1.predict(X_no_text_test)
    y_pred_with_text = model2.predict(X_with_text_test)
    print('Accuracy Score of the model without text_RidgeRegression: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_RidgeRegression: {}'.format(text_model_score))
    print('Log loss of the model with text_RidgeRegression: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_RidgeRegression: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))

def gradient_boosted_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    model = GradientBoostingClassifier(learning_rate=0.001, subsample=0.5, n_estimators=1000)
    parameter_grid = {'max_depth': [2,3,4,5,6]}
    search_model = GridSearchCV(model, parameter_grid, cv=5)
    search_model.fit(X_no_text_train,y_no_text_train)
    optimal_depth = search_model.best_estimator_.max_depth

    model1 = GradientBoostingClassifier(learning_rate=0.001, subsample=0.5, n_estimators=1000, max_depth=optimal_depth)
    model2 = GradientBoostingClassifier(learning_rate=0.001, subsample=0.5, n_estimators=1000, max_depth=optimal_depth)
    model1.fit(X_no_text_train,y_no_text_train)
    model2.fit(X_with_text_train,y_with_text_train)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)


    print('Accuracy Score of the model without text_GradientBoosted: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_GradientBoosted: {}'.format(text_model_score))
    print('Log loss of the model without text_GradientBoosted: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_GradientBoosted: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))


if __name__ == '__main__':
    data_path = 'data/Microsoft_all_data.csv'
    #logistic_regression_model(data_path)
    #ridge_regression_model(data_path)
    gradient_boosted_model(data_path)
    ridge_regression_model(data_path)
