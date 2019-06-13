from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from numpy import array
import re, pickle
import os.path


verbose     = True
checkExists = False
def print_message(*argv, end = '\n', verbose=verbose):
    if verbose:
        print(*argv, end=end)


print_message('[LOG]: Reading tokens', end='')
with open('comments.pickle', 'rb') as handle:
    comments = pickle.load(handle)
print_message(' - Done')

if os.path.isfile('model.bin') and checkExists:
    print_message('[LOG]: Model already exists, skipping training', '')
    model = Word2Vec.load('model.bin')
else:
    print_message('[LOG]: Training the Word2Vec Model', end='')
    model = Word2Vec(comments, min_count=1)
    print_message(' - DONE')
    print_message('[LOG]: Saving Model', end='')
    model.save('model.bin')
    print_message(' - DONE')

print('\n########################\n')
print('Printing Top 20 words similar to "good"')
print('\n')
print("%15s\t-" % 'Word',  'Similarity')
print("%15s\t " % '----',  '----------')
for w,p in model.most_similar(positive=['good'],topn=20):
    print("%15s\t-" % w,  p)

print('\n')
print('\n########################\n')
print('Printing Top 20 words similar to "bad"')
print('\n')
print("%15s\t-" % 'Word',  'Similarity')
print("%15s\t " % '----',  '----------')
for w,p in model.most_similar(positive=['bad'],topn=20):
    print("%15s\t-" % w,  p)

print('\n########################')

print_message('DONE')
