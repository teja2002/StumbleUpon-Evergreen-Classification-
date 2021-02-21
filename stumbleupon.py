# -*- coding: utf-8 -*-
"""

@author: Teja
"""
#importing necessary libraries  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pd
from sklearn.model_selection import cross_val_score


loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

def stumbleupon():
    train_data = list(np.array(pd.read_table('train.tsv'))[:,2])
    test_data = list(np.array(pd.read_table('test.tsv'))[:,2])
    y = np.array(pd.read_table('train.tsv'))[:,-1]  
    y=y.astype('int')   

    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                          analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

    rd = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

    X_all = train_data + test_data
    lentrain = len(train_data)

    tfv.fit(X_all)
    #transforming data
    X_all = tfv.transform(X_all)

    X = X_all[:lentrain]
    X_test = X_all[lentrain:]

    print ("Validation Score: ", np.mean(cross_val_score(rd, X, y, cv=20, scoring='roc_auc')))

    #training on the whole data
    rd.fit(X,y)
    pred = rd.predict_proba(X_test)[:,1]
    testfile = pd.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('submit.csv')
    print ("File sucessfully created!")
    print ("check your current directory for submit.csv")
    
if __name__=="__main__":
    stumbleupon()