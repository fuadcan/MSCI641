from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import metrics
import pandas as pd
import pickle, sys, re
from model_fit import model_fit

if __name__ == "__main__":
    path_list = sys.argv[1:]
    acc_uni   = model_fit((1,1),path_list)
    acc_bi    = model_fit((2,2),path_list)
    acc_unibi = model_fit((1,2),path_list)
    ##

    print('################################################')
    print('######## Calculated Test Accuracies ############')
    print('Unigram:', acc_uni)
    print('Bigram:', acc_bi)
    print('Unigram + Bigram:', acc_unibi)
    print('################################################')
