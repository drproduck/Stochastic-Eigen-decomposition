import scipy.io
import scipy.sparse as sp
import numpy as np

data = scipy.io.loadmat('20NewsHome.mat',mat_dtype=True)
fea = data['fea']
fea = sp.lil_matrix(fea)
print('converting done')
n = np.shape(fea)[0]
print(np.shape(fea))
# f = open('stop_words.txt')
f2 = open('pruned_columns','w')
columns = []
f = open('20news_stripped')
i = 0
for line in f:
    i += 1
    if not i % 5000: print('processing {} word'.format(i))
    inds = line.strip().split(' ')
    inds = [int(x) for x in inds]
    for i in inds:
        if not i == 0:
            # try:
            fea[:,inds[0]] = fea[:,inds[0]] + fea[:,i]
            # except IndexError: print(str(i))
    columns.append(inds[1:])
f.close()
scipy.io.savemat('20news_concat.mat',{'fea': fea, 'gnd': data['gnd']})
for i in columns:
    f2.write(str(i)+'\n')

# f = open('stop_words.txt')
# stop = []
# for line in f:
#     stop.append(line.strip().split(' ')[0])
#
# f2 = open('new_vocab.txt')
# vocab =
# for line in f2:
#
# for in range()


