#!/bin/python
import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np, csv,os
import matplotlib.pyplot as plt
from utils import Vocab
import pandas as pd
import pickle # load vocab
import json
import random
from collections import Counter

torch.manual_seed( 1212 )
random.seed(1212)
np.random.seed(1212)
torch.backends.cudnn.enabled = False  # need
torch.backends.cudnn.benchmark = False  # not need
torch.backends.cudnn.deterministic = True # not need


### hypermeter setting 

cache_dir = "../cache/slot/"
vocab_path = cache_dir + "vocab.pkl"
tag_map_path = cache_dir + "tag2idx.json"
data_dir = "../data/slot/"
train_path = "../data/slot/train.json"
test_path = data_dir + "test.json"
eval_path = data_dir +"eval.json"
grad_clip = 5.

onlyFirstNword = 3000


max_len = 128
num_epoch = 28

## model setting
max_seq_len = 40
batch_size = 64
hidden_size = 256
num_layers = 2
dropout = 0.2
bidirectional = True
LR = 1e-3


# test_dataset = SeqClsDataset( data_paths ,"test" )

### dataset class
# update vocabulary to containing only most frequency word
def onlyMostCommon( vocab,data, rank ):
    seqs = [ seq for seq in data.tokens.values] # list of sequencs
    c = Counter()
    for seq in seqs:
        c += Counter( seq )    # compute the freq of words
    least = len(vocab.token2idx) - rank
    least_list = c.most_common()[ : -least-1 : -1]  # least freq words
    unknownId = vocab.token2idx['[UNK]']
    for word,freq in least_list:            # update the vocab
        vocab.token2idx[word] = unknownId

class SeqSlotDataset(Dataset):
    def __init__( self,
        path ,
        mode = "train"
    ):
        self.path = path  # data
        self.data = pd.read_json( path )

        with open(  vocab_path , "rb" ) as f:  # vocab
            self.vocab = pickle.load(f)

        if mode =="train" and onlyFirstNword is not None:
            print( f"update vocab for reserving first {onlyFirstNword} words" )
            onlyMostCommon( self.vocab, self.data, onlyFirstNword )

        with open( tag_map_path, 'r') as f:
            self.tag2idx = json.load(f)
        self.tag2idx['pad'] = 9
        self.idx2tag = { v:k  for k,v in self.tag2idx.items() }

        self.mode = mode
        self._initial()

    def _initial(self):
        text_data = self.data.tokens.values
        # 各句子長度 --> tensor
        seq_len = [ len(seq) for seq in text_data ]
        self.seq_len = torch.tensor( seq_len )
        # text --> token  + pading (最長句子 35) --> tensor
        seqs = self.vocab.encode_batch(  text_data, to_len= max_seq_len  )
        self.seqs = torch.tensor( seqs )
        # tag2idx --> tensor
        if self.mode == "train" or self.mode == "eval":
            self.make_tags_token( self.data.tags.values )
        

    def make_tags_token(self,tags_batch, to_len= max_seq_len ):
        self.tags = []
        for i,tags in enumerate(tags_batch):
            tag_token = [ self.tag2idx[tag] for tag in tags]
            tag_token += [ self.tag2idx['pad'] ] * ( to_len - self.seq_len[i] )  # padding
            self.tags.append( tag_token )
        self.tags = torch.tensor( self.tags )
            
            

    def __len__(self) -> int:
        return len( self.data.index )

    def __getitem__(self, index) :
        if self.mode == "train" or self.mode == "eval" :
            return ( self.seqs[index], self.seq_len[index]), self.tags[index]
        else :
            return self.data.id[index],(self.seqs[index],self.seq_len[index])

    @property
    def num_classes(self) -> int:
        return len(self.tag2idx)

    def collate_fn(self, batch ):
        pass

    # def label2idx(self, label: str):
    #     return self.label_mapping[label]
    def idx2label(self, idx: int):
        return self._idx2label[idx]
    def batch_idx2label(self,batch, seq_len):
        batch = batch.tolist()  # tensor --> int list
        classes = []
        for seq, length in zip(batch,seq_len):
            cls = [ self.idx2tag[ seq[idx] ] for idx in range(length) ]
            classes += [cls]
            # classes = [  self.idx2label(num) for num in cls_num ]
        return classes

# make dataset
train_set = SeqSlotDataset( train_path, "train" )
test_set = SeqSlotDataset( test_path, "test" )
eval_set = SeqSlotDataset( eval_path , "eval" )

train_loader = DataLoader( train_set, batch_size= batch_size, shuffle=True )
eval_loader = DataLoader( eval_set, batch_size= batch_size, shuffle=False )
test_loader = DataLoader( test_set, batch_size= batch_size, shuffle=False )


### model class + score class

class TagMultiClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
        emb_path =  cache_dir + "embeddings.pt"
        ) -> None:
        super(TagMultiClassifier, self).__init__()

        # TODO: model architecture
        self.embed = self.loadembedding( emb_path )

        self.input_size = self.embed.weight.shape[1]
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.bidirectional = bidirectional
        self.dropout = dropout 
        self.num_classes = num_classes

        self.rnn = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                dropout = dropout,
                bidirectional = self.bidirectional,
                batch_first = True
                )
        # pooling (batch,hidden_size,embeddings) -->(batch,1,embeddings)
        self.fc = nn.Sequential(
                nn.Linear( self.encoder_output_size, 256 ),
                nn.Dropout( self.dropout ),
                nn.Linear( 256, self.num_classes),
                nn.SiLU()
                )

    @property
    def encoder_output_size(self) -> int:
        if self.bidirectional:
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def loadembedding( self,emb_path ):
        # removeLessWord( emb_path )
        embeddings = torch.load( emb_path )
        embeddings = Embedding.from_pretrained( embeddings, freeze=False)
        return embeddings

    def forward(self, x_set ):
        x , seq_len = x_set
        x = self.embed( x )
        x = pack_padded_sequence(x, seq_len ,batch_first=True,enforce_sorted=False)
        out,(h,_) = self.rnn( x )

        out,out_len = pad_packed_sequence( out,batch_first=True)
        # out = out.view( -1, self.encoder_output_size ) # concate each seq
        predict = self.fc( out )
        # ( batch_size, seq_len, num_classes ) --> 
        # ( batch_size, num_classes, seq_len )
        predict = predict.permute( 0, 2, 1 )

        # padding seq for calculas crossEntrophy
        # pad_len = ( 0 , max_seq_len - predict.size(-1) )
        # predict = nn.functional.pad( predict, pad_len )

        return predict
        
    def get_embeded_packed_batch(self, batch ):
        embed_batch = self.embed( batch.data )
        packed_embed_batch = PackedSequence( embed_batch, 
                batch.batch_sizes, 
                sorted_indices=batch.sorted_indices, 
                unsorted_indices=batch.unsorted_indices )
        return packed_embed_batch

from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

from livelossplot import PlotLosses
liveloss = PlotLosses()
log = {}

class Score():
    def __init__(self, criterion ,pbar = None):
        self.total_loss = 0
        self.total_hit = 0
        self.num = 0
        self.total_seq_len = 0 
        self.criterion = criterion
        self.pbar = pbar
        pass
    def eval(self,predicts,y):
        pass
    def __call__(self, predicts,y ,seq_len ):
        loss = self.criterion( predicts, y )
        self.num += len( y )

        predicts = torch.argmax(predicts,dim=1).squeeze()
        hit = 0
        for i,num in enumerate( seq_len ):
            # hit += (predicts[i][:num] == y[i][:num] ).sum()
            if torch.equal( predicts[i][:num], y[i][:num] ):
                hit += 1
        # hit = (predicts == y).sum()
        # seq_len = sum( seq_len )
        # acc = hit / seq_len * 100
        self.total_hit += hit
        acc = self.total_hit / self.num * 100

        # self.total_seq_len += seq_len
        self.total_loss += loss

        message =  f"loss:{loss:.2f} acc:{acc:.2f}%" 
        if self.pbar:
            self.pbar.set_postfix_str( message )
        return loss

    def show_score(self , epoch ):
        # avg_acc = self.total_hit / self.total_seq_len * 100
        avg_acc = self.total_hit / self.num * 100
        print( f"e {epoch} avg_acc:{avg_acc:.2f}  loss:{self.total_loss:.2f}                        " )
        log["loss"] = self.total_loss.item()
        # log["acc"] = avg_acc.item()
        log["acc"] = avg_acc
        # liveloss.update( log )
        # liveloss.send()

    def show_class_report( self,predicts,y):
        predicts = torch.argmax(predicts,1).squeeze() 
        print( classification_report( y,predicts,mode="strict",scheme=IOB2) )


# y_true = [['B-PER','K-AA'], ['B-PER', 'I-PER', 'O']]
# y_pred = [['B-PER','K-AA'], ['B-PER', 'I-PER', 'O']]
# print( classification_report( y_true,y_pred, mode = "strict")) 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TagMultiClassifier( hidden_size,
        num_layers,
        dropout,
        bidirectional,
        train_set.num_classes ).to(device)

optimizer  = torch.optim.Adam( model.parameters(), lr=LR )
criterion = nn.CrossEntropyLoss( ignore_index = train_set.tag2idx['pad'] )

### training

from tqdm import tqdm,trange


def train(model,dataloader, epoch):
    model.train()
    pbar = tqdm(dataloader)
    metrics = Score( criterion ,pbar )
    for (x,seq_len),y in pbar:
        x = x.to(device)
        y = y.to(device)

        predicts = model( (x,seq_len) )
        max_seq_len = predicts.size(2)
        y = y[:,:max_seq_len]
        # loss = criterion( predicts, y )
        loss = metrics( predicts, y, seq_len )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= grad_clip)

        optimizer.step()
    metrics.show_score( epoch+1 )
    # print( f"epoch: {epoch+1} loss:{loss} accuracy:{accuracy}" )



@torch.no_grad()
def eval(model,dataloader,epoch):
    model.eval()
    metrics = Score( criterion  )
    for (x,seq_len),y in dataloader:
        x = x.to(device)
        y = y.to(device)

        predicts = model( (x,seq_len) )
        max_seq_len = predicts.size(2)
        y = y[:,:max_seq_len]
        loss = metrics( predicts, y , seq_len)
    metrics.show_score( epoch+1  )
    print(  )


def main():
    for epoch in range( num_epoch ):
        train( model, train_loader , epoch)
        eval( model, eval_loader ,epoch)
    test( model, test_loader)


import csv
from pathlib import Path
from datetime import datetime
def get_now_time_str():
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d--%H-%M-%S")
    return time_str

def write_result(result_list):
    result_dir = "../result/slot/"
    Path( result_dir ).mkdir( parents=True, exist_ok=True)
    result_path = result_dir  + get_now_time_str() + '.csv' 
    with open( result_path ,'w',newline='') as f:
        w = csv.writer( f )
        w.writerow(["id","tags" ])               
        w.writerows( result_list )
    print( f"write to {result_path} completed" )

@torch.no_grad()
def test(model, dataloader):
    model.eval()
    result_list = []
    for test_ids , (x,seq_len) in dataloader:
        x = x.to(device)
        predicts = model( (x,seq_len) )
        predicts = torch.argmax( predicts , dim=1 ).squeeze()
        predicts = dataloader.dataset.batch_idx2label( predicts,seq_len )
        result_batch = [ [i," ".join(j) ] for i,j in zip(test_ids,predicts) ]
        result_list += result_batch
    write_result( result_list )

if __name__ == "__main__":
    main()
