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
from extra_cleaning_obj1 import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as sm
import pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import recall_score, roc_curve, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier





def logistic_regression_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_logistic_sub(data_path, n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    model1 = LogisticRegression()
    model2 = LogisticRegression()
    model1.fit(X_no_text_train,y_no_text_train)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    model2.fit(X_with_text_train,y_with_text_train)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)

    print('-------LOGISTIC REGRESSION-------')
    print('Accuracy Score of the model without text_LogisticRegression: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_LogisticRegression: {}'.format(text_model_score))
    print('Log loss of the model without text_LogisticRegression: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_LogisticRegression: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without textLogisticRegression: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_LogisticRegression: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_LogisticRegression: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_LogisticRegression: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_LogisticRegression: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_LogisticRegression: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_LogisticRegression: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_LogisticRegression: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))



def gradient_boosted_model(data_path, i):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

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

    fpr_no_text, tpr_no_text, _ = roc_curve(y_no_text_test, y_pred_no_text)
    roc_auc_no_text = auc(fpr_no_text, tpr_no_text)
    fpr_with_text, tpr_with_text, _ = roc_curve(y_with_text_test, y_pred_with_text)
    roc_auc_with_text = auc(fpr_with_text, tpr_with_text)
    plt.figure()
    plt.plot(fpr_with_text, tpr_with_text, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_with_text)
    plt.plot(fpr_no_text, tpr_no_text, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % oc_auc_no_text)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    fig_name = 'auc_results/fig_' + i
    plt.savefig(fig_name)




    print('-------GRADIENT BOOSTED-------')
    print('Accuracy Score of the model without text_GradientBoosted: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_GradientBoosted: {}'.format(text_model_score))
    print('Log loss of the model without text_GradientBoosted: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_GradientBoosted: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without text_GradientBoosted: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_GradientBoosted: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_GradientBoosted: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_GradientBoosted: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_GradientBoosted: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_GradientBoosted: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_GradientBoosted: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_GradientBoosted: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))



    return model2

def random_forest_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    model1 = RandomForestClassifier(max_depth=5, min_samples_split=3, n_jobs=-1)
    model2 = RandomForestClassifier(max_depth=5, min_samples_split=3, n_jobs=-1)

    model1.fit(X_no_text_train,y_no_text_train)
    model2.fit(X_with_text_train,y_with_text_train)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)


    print('-------RANDOM FOREST-------')
    print('Accuracy Score of the model without text_RandomForest: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_RandomForest: {}'.format(text_model_score))
    print('Log loss of the model without text_RandomForest: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_RandomForest: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without text_RandomForest: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_RandomForest: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_RandomForest: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_RandomForest: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_RandomForest: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_RandomForest: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_RandomForest: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_RandomForest: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))

def decision_tree_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    model1 = DecisionTreeClassifier(max_depth=4, min_samples_split=3)
    model2 = DecisionTreeClassifier(max_depth=4, min_samples_split=3)

    model1.fit(X_no_text_train,y_no_text_train)
    model2.fit(X_with_text_train,y_with_text_train)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)


    print('-------DECISION TREE-------')
    print('Accuracy Score of the model without text_DecisionTree: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_DecisionTree: {}'.format(text_model_score))
    print('Log loss of the model without text_DecisionTree: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_DecisionTree: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without text_DecisionTree: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_DecisionTree: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_DecisionTree: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_DecisionTree: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_DecisionTree: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_DecisionTree: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_DecisionTree: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_DecisionTree: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))


def NaiveBayes_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    model1 = MultinomialNB(alpha=0.1)
    model2 = MultinomialNB(alpha=0.1)

    model1.fit(X_no_text_train,y_no_text_train)
    model2.fit(X_with_text_train,y_with_text_train)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)

    print('-------NAIVE BAYES-------')
    print('Accuracy Score of the model without text_NaiveBayes: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_NaiveBayes: {}'.format(text_model_score))
    print('Log loss of the model without text_NaiveBayes: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_NaiveBayes: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without text_NaiveBayes: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_NaiveBayes: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_NaiveBayes: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_NaiveBayes: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_NaiveBayes: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_NaiveBayes: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_NaiveBayes: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_NaiveBayes: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))



def adaboost_model(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    model1 = AdaBoostClassifier(learning_rate=0.0001)
    model2 = AdaBoostClassifier(learning_rate=0.0001)

    model1.fit(X_no_text_train,y_no_text_train)
    model2.fit(X_with_text_train,y_with_text_train)
    y_pred_no_text = model1.predict_proba(X_no_text_test)
    y_pred_with_text = model2.predict_proba(X_with_text_test)
    no_text_model_score = model1.score(X_no_text_test, y_no_text_test)
    text_model_score = model2.score(X_with_text_test,y_with_text_test)

    print('-------ADABOOST-------')
    print('Accuracy Score of the model without text_NaiveBayes: {}'.format(no_text_model_score))
    print('Accuracy Score of the model with text_NaiveBayes: {}'.format(text_model_score))
    print('Log loss of the model without text_NaiveBayes: {}'.format(log_loss(y_no_text_test, y_pred_no_text)))
    print('Log loss of the model with text_NaiveBayes: {}'.format(log_loss(y_with_text_test, y_pred_with_text)))
    print('f1_score of the model without text_NaiveBayes: {}'.format(f1_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('f1_score of the model text_NaiveBayes: {}'.format(f1_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('precision_score of the model without text_NaiveBayes: {}'.format(precision_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('precision_score of the model text_NaiveBayes: {}'.format(precision_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('recall_score of the model without text_NaiveBayes: {}'.format(recall_score(y_no_text_test, model1.predict(X_no_text_test))))
    print('recall_score of the model text_NaiveBayes: {}'.format(recall_score(y_with_text_test, model2.predict(X_with_text_test))))
    print('roc_auc_score of the model without text_NaiveBayes: {}'.format(roc_auc_score(y_no_text_test, y_pred_no_text[:,1].reshape(-1,1))))
    print('roc_auc_score of the model text_NaiveBayes: {}'.format(roc_auc_score(y_with_text_test, y_pred_with_text[:,1].reshape(-1,1))))





def explore_logistic_regression(data_path):
    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = \
                train_test_data_splitting_after_KMeans_sub(data_path,  n_groups=6)

    X_no_text_train = X_no_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_no_text_test = X_no_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_train = X_with_text_train.drop(columns = ['Unnamed: 0.1'], axis=1)
    X_with_text_test = X_with_text_test.drop(columns = ['Unnamed: 0.1'], axis=1)

    X_no_text = pd.concat([X_no_text_test, X_no_text_train], axis = 0)
    X_with_text = pd.concat([X_with_text_test, X_with_text_train], axis = 0)
    y_no_text = pd.concat([y_no_text_test, y_no_text_train], axis = 0)
    y_with_text = pd.concat([y_with_text_test, y_with_text_train], axis = 0)


    model1 = sm.Logit(y_no_text, X_no_text)
    model2 = sm.Logit(y_with_text, X_with_text)
    result1= model1.fit()
    result2= model2.fit()
    print('No text')
    print(result1.summary())
    print('-------------------------------')
    print('With text')
    print(result2.summary())

if __name__ == '__main__':
    data_path = 'temp_data/Feb25_data.csv'
    i = 0
    #explore_logistic_regression(data_path)
    while i <100:
        print('-------------------------------')
        print('Model number {}'.format(i+1))
        logistic_regression_model(data_path)
        # gbcmodel = gradient_boosted_model(data_path)
        #print(len(pd.read_csv(data_path)))
        gradient_boosted_model(data_path)
        # with open('website/gbcmodel.pkl', 'wb') as f:
        #     pickle.dump(gbcmodel, f)
        random_forest_model(data_path)
        decision_tree_model(data_path)
        adaboost_model(data_path)
        NaiveBayes_model(data_path)
        print('-------------------------------')
        print('-------------------------------')
        i += 1
