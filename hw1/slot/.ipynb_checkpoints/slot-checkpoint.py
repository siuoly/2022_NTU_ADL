#!/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, csv,os
import matplotlib.pyplot as plt
from utils import Vocab
import pandas as pd
import pickle # load vocab
import json


###
cache_dir = "../cache/slot/"
vocab_path = cache_dir + "vocab.pkl"
tag_map_path = cache_dir + "tag2idx.json"
data_dir = "../data/slot/"
train_path = "../data/slot/train.json"
test_path = data_dir + "test.json"
eval_path = data_dir +"eval.json"

max_len = 128
num_epoch = 10


# test_dataset = SeqClsDataset( data_paths ,"test" )

###
class SeqSlotDataset(Dataset):
    def __init__( self,
        path ,
        mode = "train"
    ):
        self.path = path  # data
        self.data = pd.read_json( path )

        with open(  vocab_path , "rb") as f:  # vocab
            self.vocab = pickle.load(f)
        with open( tag_map_path, 'r') as f:
            self.tag2idx = json.load(f)
        self.tag2idx['pad'] = 9

        self.mode = mode
        self._initial()

    def _initial(self):
        # import pdb; pdb.set_trace()
        text_data = self.data.tokens.values
        # 各句子長度 --> tensor
        seq_len = [ len(seq) for seq in text_data ]
        self.seq_len = torch.tensor( seq_len )
        # text --> token  + pading (最長句子 35) --> tensor
        seqs = self.vocab.encode_batch(  text_data, to_len=40  )
        self.seqs = torch.tensor( seqs )
        # tag2idx --> tensor
        if self.mode == "train":
            self.make_tags_token( self.data.tags.values )
        

    def make_tags_token(self,tags_batch, to_len=40 ):
        self.tags = []
        for i,tags in enumerate(tags_batch):
            tag_token = [ self.tag2idx[tag] for tag in tags]
            tag_token += [ self.tag2idx['pad'] ] * ( to_len - self.seq_len[i] )  # padding
            self.tags.append( tag_token )
        self.tags = torch.tensor( self.tags )
            
            

    def __len__(self) -> int:
        return len( self.data.index )

    def __getitem__(self, index) :
        if self.mode == "train":
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
    # def idx2label(self, idx: int):
    #     return self._idx2label[idx]
    # def batch_idx2label(self,batch):
    #     cls_num = batch.tolist()  # tensor --> int list
    #     classes = [  self.idx2label(num) for num in cls_num ]
    #     return classes

### make dataset
train_set = SeqSlotDataset( train_path, "train" )
test_set = SeqSlotDataset( test_path, "test" )

train_loader = DataLoader( train_set, batch_size=128, shuffle=True )
test_loader = DataLoader( test_set, batch_size=128, shuffle=False )


### make module

class TagMultiClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
        emb_path = "../cache/slot/embeddings.pt"
        ) -> None:
        super(SeqClassifier, self).__init__()

        # TODO: model architecture
        self.embed = self.loadembedding( emb_path )

        self.input_size = embeddings.size(1)
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
        self.pool = nn.MaxPool2d( (128,1) ) # padding seq_len=128

        self.fc = nn.Sequential(
                nn.Dropout(dropout) ,
                nn.Linear( self.encoder_output_size, self.num_classes),
                nn.SiLU()
                )

    @property
    def encoder_output_size(self) -> int:
        if self.bidirectional:
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def loadembedding( self.emb_path ):
        embedding_path = "./cache/intent/embeddings.pt"
        embeddings = torch.load( embedding_path )
        return embeddings

    def forward(self, x_set ):
        x , seq_len = x_set
        x = self.embed( x )
        x = pack_padded_sequence(x, seq_len ,batch_first=True,enforce_sorted=False)
        out,(h,_) = self.rnn( x )
        out,out_len = pad_packed_sequence( out,batch_first=True)
        
        if self.bidirectional:
            h = torch.cat( (h[-1],h[-2]), axis=-1)
        else:
            h = h[-1]
        out = self.fc( h )

        # import ipdb; ipdb.set_trace()
        # pool = nn.MaxPool2d( (max(seq_len),1 ))
        # out = pool(out)
        # out = self.fc( out )
        # out = out.squeeze(1)

        return out

    def get_embeded_packed_batch(self, batch ):
        embed_batch = self.embed( batch.data )
        packed_embed_batch = PackedSequence( embed_batch, 
                batch.batch_sizes, 
                sorted_indices=batch.sorted_indices, 
                unsorted_indices=batch.unsorted_indices )
        return packed_embed_batch

###
class Score():
    def __init__(self):
        self.total_loss = 0
        self.total_acc = 0
        pass
    def eval(self,predicts,y):
        pass
    def __call__(self,predicts,y):
        print( 123123 )
        pass
###

model = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = 
criteria = 

device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm,trange
def train(model,dataloader,num_epoch):
    for i in num_epoch:
        metrics = Score()
        for (x,seq_len),y in tqdm(dataloader):
            x = x.to(device)
            seq_len = seq_len.to(device)
            y = y.to(device)

            predicts = model( x,seq_len )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss,accuracy = metrics( predicts,y )
        print( f"epoch: {epoch+1} loss:{loss} accuracy:{accuracy}" )


def eval(model,dataloader,num_epoch):
    for i in num_epoch:
        metrics = Score()
        for (x,seq_len),y in tqdm(dataloader):
            x = x.to(device)
            seq_len = seq_len.to(device)
            y = y.to(device)

            predicts = model( x,seq_len )
            loss,accuracy = metrics( predicts,y )
        print( f"epoch: {epoch+1} loss:{loss} accuracy:{accuracy}" )


### debug using
# "../data/slot/test.json"
data = pd.read_json( train_path )


