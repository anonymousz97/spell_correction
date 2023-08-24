import re
import unicodedata
from collections import Counter
from itertools import chain
from utils import check_number_in_word
noise = []
corpus = []
for line in open("data_lower.txt"):
    sent = line.strip().split()
    corpus.append(sent)

train = dict(Counter(chain(*corpus)))
del(corpus)
vocab = []
for line in open("vocab_lower.txt"):
    sent = line.strip().split()
    vocab.append(sent)

vocab= dict(Counter(chain(*vocab)))
k = sorted(train, key=lambda w: train[w], reverse=True)
for i in k:
    try:
        int(i)
    except:
        if i not in vocab and not check_number_in_word(i):
            if train[i] >=10 and not i.isupper():
                noise.append(i)
print(len(noise))
noise = list(vocab.keys()) + noise
del(train)
del(vocab)
del(k)
print(len(noise))
j = 0
f = open("vocab_lower.txt", "w")
for i in range(len(noise)):
    j+=1
    f.write(noise[i] + "\n")
    if j == 20000:
        break
f.close()
