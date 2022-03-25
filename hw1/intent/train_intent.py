import random
import numpy as np
import torch
torch.manual_seed( 1212 )
random.seed(1212)
np.random.seed(1212)
torch.backends.cudnn.enabled = False  # need
torch.backends.cudnn.benchmark = False  # not need
torch.backends.cudnn.deterministic = True # not need

import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict


from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange,tqdm

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier,ACC,save_model,load_model

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]



def get_datasets(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    train_path = args.data_dir / "train.json"
    eval_path = args.data_dir/ "eval.json"
    train_data = json.loads( train_path.read_text()  )
    eval_data = json.loads( eval_path.read_text() )

    datasets = dict()
    datasets['train'] = SeqClsDataset( train_data , vocab, intent2idx, args.max_len)
    datasets['eval' ] = SeqClsDataset( eval_data , vocab, intent2idx, args.max_len)


    return datasets

def train_epoch(model,dataloader,optimizer,loss_func):
    import pdb; pdb.set_trace()
    model.train()
    # Accuracy
    total_loss, total ,hit = 0 ,0,0
    # TODO: Training loop - iterate over train dataloader and update model weights
    # import pdb; pdb.set_trace()
    for (x,seq_len), y in tqdm(dataloader):
        x = x.to(args.device)
        y = y.to(args.device)
        optimizer.zero_grad()
        pred_y = model( (x,seq_len) )
        loss = loss_func( pred_y,y )
        loss.backward()
        optimizer.step()

        # Evaluate: Accuracy: hit/total
        _,pred_class = torch.max(pred_y,1)
        total += len(y)
        hit += (pred_class == y).sum().item()
        total_loss += float(loss)

    accuracy = hit/total*100
    print( f"train:loss:{ total_loss }, accuracy:{ accuracy }" )


@torch.no_grad()
def eval_epoch(model,eval_loader,optimizer,loss_func):
    model.eval()
    total_loss, total ,hit = 0 ,0,0
    # TODO: Training loop - iterate over train dataloader and update model weights
    for (x,seq_len), y in eval_loader :
        x = x.to( args.device )
        y = y.to( args.device )
        pred_y = model( (x,seq_len) )
        loss = loss_func( pred_y,y )

        # Evaluate: Accuracy: hit/total
        _,pred_class = torch.max(pred_y,1)
        total += len(y)
        hit += (pred_class == y).sum().item()
        total_loss += float(loss)

    accuracy = hit/total*100
    print( f"eval: loss:{ total_loss }, accuracy:{ accuracy }" )
    return loss,accuracy


from test_intent import test, get_testloader



def main(args):
    # TODO: crecate DataLoader for train / dev datasets
    device = args.device

    datasets = get_datasets(args)
    train_dataset = datasets['train']
    eval_dataset = datasets['eval']

    dataloader = DataLoader(  train_dataset, args.batch_size, 
            shuffle = True )# ,collate_fn = train_dataset.collate_fn)
    eval_loader = DataLoader(  eval_dataset, args.batch_size, 
            shuffle = False)# ,collate_fn = train_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(embeddings, 
            args.hidden_size, 
            args.num_layers, 
            args.dropout, 
            args.bidirectional,
            train_dataset.num_classes).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    import pdb; pdb.set_trace()

    preacc = 0
    test_loader = get_testloader( args.batch_size )
    for epoch in range(1, args.num_epoch + 1):
        print(  f"epoch:{epoch}" )
        train_epoch(model,dataloader,optimizer,loss_func)
        loss,accuracy = eval_epoch(model, eval_loader,optimizer,loss_func)

        # if accuracy > preacc:
        #     preacc = accuracy
        save_model(model, f"E-{epoch}__" )
        if epoch > 0:
            test( model , test_loader )



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="../cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="../ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)
    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    # scheduler # TODO
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=20)
    # seed,wandb ...
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    clip_grad = 0.5

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
