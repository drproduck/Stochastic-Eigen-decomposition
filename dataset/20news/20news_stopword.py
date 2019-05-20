import scipy.io
import scipy.sparse as sp
import numpy as np

f_stop_word_name = open('word_name','w')

f = open('stop_words.txt')
stop_words = []
for line in f:
    word = line.strip().split(' ')[0]
    stop_words.append(word)
f.close()
print('done getting stop words')

f = open('vocabulary.txt')
vocab = set()
stop_word_inds = []
i = 1
for line in f:
    if not (i-1) % 5000: print('processing word {} to {}'.format(i, i + 4999))
    word = line.strip().split(' ')[0]
    if word in vocab: raise Warning('Duplicate: word {} already in vocab'.format(word))
    vocab.add(word)
    if word in stop_words:
        stop_word_inds.append(i)
        f_stop_word_name.write(word+'\n')
    i += 1
f.close()
print('total stop words = {}, removed {}, flushing to file ...'.format(len(stop_words), len(stop_word_inds)))

f = open('stop_word_inds','w')
for it in stop_word_inds:
    f.write(str(it)+'\n')
f.flush(); f.close()
f_stop_word_name.flush(); f_stop_word_name.close()
