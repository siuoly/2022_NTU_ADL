#!/bin/python
import json
import torch
import pandas as pd
import pickle # load vocab
import utils
from collections import Counter
###
data_dir = "../data/slot/"
train_path = data_dir + "train.json"
vocab_path = "../cache/slot/vocab.pkl"

data = pd.read_json( train_path )
with open(  vocab_path , "rb") as f:  # vocab
    vocab = pickle.load(f)
###
# get total word data, and encode
seqs = [ seq for seq in data.tokens.values]
###
# counter create
c = Counter()
for seq in seqs:
    c += Counter( seq )
###
#　list the rank
def leastCommon(c,n):
    return c.most_common()[:-n-1:-1]
least_list = leastCommon( c, 1000 )
###
for word,freq in least_list:
    print( word )
###

# create a list of word for less frequency
def onlyMostCommon( vocab,data, rank ):
    seqs = [ seq for seq in data.tokens.values]
    c = Counter()
    for seq in seqs:
        c += Counter( seq )
    least_list = c.most_common()[ : -rank-1 : -1] 
    unknownId = vocab.token2idx['[UNK]']

    for word,freq in least_list:
        vocab.token2idx[word] = unknownId
onlyMostCommon( vocab, data, 3000)

###

###
vocab.token2idx


# map the word the <unk>
# update vocab


