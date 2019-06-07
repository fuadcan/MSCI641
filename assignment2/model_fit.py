from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import metrics
import pandas as pd
import pickle, sys, re




# ngram_range   = (1,1)
# verbose = True
path_list = ['assignment2/neg_test.pickle',
'assignment2/neg_train.pickle',
'assignment2/neg_val.pickle',
'assignment2/pos_test.pickle',
'assignment2/pos_train.pickle',
'assignment2/pos_val.pickle']

##

def model_fit(ngram_range,path_list,verbose=True):
    def print_message(*argv, end = '\n', verbose=verbose):
        if verbose:
            print(*argv, end=end)


    if sum(ngram_range) == 2:
        print_message('[LOG]: Learning over Unigrams')
    elif sum(ngram_range) == 3:
        print_message('[LOG]: Learning over Unigram+Bigrams')
    elif sum(ngram_range) == 4:
        print_message('[LOG]: Learning over Bigrams')
    else:
        print_message('[LOG]: Custom ngram_range:', ngram_range)


    print_message('[LOG]: Reading tokens ', end = '')

    token_dict = dict()
    for path in path_list:
        key = re.findall('pos|neg',path)[0] + '_' + re.findall('train|test|val',path)[0]
        with open(path, 'rb') as handle:
            token_dict[key] = pickle.load(handle)

    X_train = token_dict['pos_train'] + token_dict['neg_train']
    X_test  = token_dict['pos_test']  + token_dict['neg_test']
    X_val   = token_dict['pos_val']   + token_dict['neg_val']

    print_message('- DONE')

    ## Flag tokens
    print_message('[LOG]: Flagging tokens to input sklearn package ')
    X_train = ['<<' + '>><<'.join(comment) + '>>' for comment in X_train]
    X_test  = ['<<' + '>><<'.join(comment) + '>>' for comment in X_test]
    X_val   = ['<<' + '>><<'.join(comment) + '>>' for comment in X_val]

    ##
    print_message('[LOG]: Inputting tokens to sklearn', end = '')
    ## Input to sklearn
    vect = CountVectorizer(token_pattern='<<(.*?)>>',ngram_range=ngram_range)
    vect.fit(X_train)
    print_message('- DONE')

    print_message('[LOG]: Creating X and y ', end = '')
    X_train_mat = vect.transform(X_train)
    X_test_mat  = vect.transform(X_test)
    X_val_mat   = vect.transform(X_val)

    y_train = pd.Series([1]*len(token_dict['pos_train']) + [0]*len(token_dict['neg_train']))
    y_test  = pd.Series([1]*len(token_dict['pos_test'])  + [0]*len(token_dict['neg_test']))
    y_val   = pd.Series([1]*len(token_dict['pos_val'])   + [0]*len(token_dict['neg_val']))

    print_message('- DONE')

    def calculate_accuracy(alpha):
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train_mat,y_train)
        y_pred_class = nb.predict(X_val_mat)
        print_message('[alpha=',alpha/100,']',metrics.accuracy_score(y_val, y_pred_class))
        # print(metrics.f1_score(y_val, y_pred_class))
        return(metrics.accuracy_score(y_val, y_pred_class))

    print_message('[LOG]: Tuning alpha over validation set')
    val_results = [(alpha,calculate_accuracy(alpha)) for alpha in range(1,101)]
    alpha_max = [alpha/100 for alpha, acc in val_results if acc == max([acc for a,acc in val_results])][0]
    print_message('- DONE')

    nb = MultinomialNB(alpha=alpha_max)
    nb.fit(X_train_mat,y_train)
    y_pred_class = nb.predict(X_test_mat)
    print_message('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    print_message('F1 score:', metrics.f1_score(y_test, y_pred_class))
    print_message('[LOG]: DONE')
    return(metrics.accuracy_score(y_test, y_pred_class))
