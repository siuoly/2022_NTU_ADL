from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
        ) -> None:
        super(SeqClassifier, self).__init__()

        # TODO: model architecture
        self.embed = Embedding.from_pretrained(embeddings, freeze=False )

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
                batch_first = False
                )
        # pooling (batch,hidden_size,embeddings) -->(batch,1,embeddings)
        
        # self.pool = nn.MaxPool2d( (128,1) ) # padding seq_len=128


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

        return out

    def get_embeded_packed_batch(self, batch ):
        embed_batch = self.embed( batch.data )
        packed_embed_batch = PackedSequence( embed_batch, 
                batch.batch_sizes, 
                sorted_indices=batch.sorted_indices, 
                unsorted_indices=batch.unsorted_indices )
        return packed_embed_batch



# from dataset import get_datasets
def get_embedding():
    embedding_path = "./cache/intent/embeddings.pt"
    embeddings = torch.load( embedding_path )
    embeddings = Embedding.from_pretrained(embeddings, freeze=False )
    return embeddings


def show_dataset():
    print( ds, tds )
    print( f"{embeddings=}" ) 
    print( embeddings ) 


import dataset
def get_vocab():
    from utils import Vocab
    cache_dir = "../cache/intent/"
    with open( cache_dir + "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    return vocab

def get_labels(batch):
    pass

def ACC(pred_y,y):
    _,pred_class = torch.max(pred_y,1)
    total = len(y)
    hit = (pred_class == y).sum().item()
    accuracy = hit/total * 100
    return accuracy


from datetime import datetime
def get_now_time_str():
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d--%H-%M-%S")
    return time_str

from pathlib import Path
def save_model(model, model_name =""):
    # 確保model資料夾存在
    Path( model_dir ).mkdir( parents=True, exist_ok=True)
    model_dir = "../model" 
    torch.save( model.state_dict(), 
            model_dir + model_name + get_now_time_str() + "SeqClassifier.ckpt" )

def load_model(model, model_path):
    model.load_state_dict( torch.load(model_path) )

if __name__ == "__main__":
    from train_intent import parse_args
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = dataset.get_datasets()[0]['train']
    # show_dataset()

    from torch.utils.data import Dataset, DataLoader
    dataloader = DataLoader(train_set,
            batch_sizes = args.batch_sizes,
            shuffle=True,
            collate_fn = train_set.collate_fn )

    embeddings = get_embedding()
    model = SeqClassifier(embeddings, args.hidden_size, 
            args.num_layers, args.dropout, 
            args.bidirectional, train_set.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()


    from tqdm import tqdm
    for i in range(50):
        load = tqdm(dataloader)
        for (x,seq_len), y in load:
            optimizer.zero_grad()
            x = x.to(device)
            pred_y = model( (x,seq_len) )
            y = y.to(device)
            loss = loss_func( pred_y,y )
            loss.backward()
            optimizer.step()
        print( f"loss:{float(loss)}, accuracy:{ACC(pred_y,y)}" )


