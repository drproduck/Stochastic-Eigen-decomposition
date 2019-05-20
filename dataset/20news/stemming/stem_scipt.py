#!/usr/bin/env python
from nltk.stem.snowball import *

#stem words from 20news
#repeat_inds: index of repeated word after stemming (optional)
#new_vocab: list of words after stemming
#new_vocab_word_to_index: map from index of word in new_vocab to list of indices of repeated words in previous vocabulary
#save to 20news_indicator_matrix: matlab sparse matrix follows:
# row column 1
# row is index of word in new_vocab, column is index of repeated word corresponding to word after stemming

stemmer = SnowballStemmer('english', ignore_stopwords=False)
f = open('vocabulary.txt')
vocab = []
for line in f:
    vocab.append(line.strip().split(' ')[0])
print('number of words: '+str(len(vocab)))
new_vocab = []
new_vocab_word_to_index = {}
repeat_inds = []

for i in range(len(vocab)):
    if not i % 5000:
        print('process word {} to word {}'.format(i, (i+5000)))
    word = vocab[i]
    w = stemmer.stem(word)
    if w in new_vocab:
        repeat_inds.append(i)
        new_vocab_word_to_index[w].append(i)
        if i < 500:
            print('word {}, repeater {}, index {}'.format(w, word, i))
    else:
        new_vocab.append(w)
        new_vocab_word_to_index[w] = [i]
sfile = open('new_vocab','w')
print('saving...')
for word in new_vocab:
    sfile.write(word+'\n')
sfile.close()
sfile = open('20news_indicator_matrix','w')

#NOTE: when write, index should starts from 1 to suit matlab
row = 1
for idx_list in new_vocab_word_to_index.values():
    for i in idx_list:
        sfile.write(str(row)+' '+str(i+1)+' '+str(1))
        sfile.write('\n')
    row = row + 1

print('strip {} words, remaining {}'.format(len(repeat_inds), len(new_vocab)))