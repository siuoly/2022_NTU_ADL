import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from model import SeqClassifier,ACC,save_model,load_model
from torch.utils.data import Dataset, DataLoader
from dataset import SeqClsDataset

import random
import numpy as np
torch.manual_seed( 1212 )
random.seed(1212)
np.random.seed(1212)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test( model, dataloader ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_list = []
    for test_id, (x,seq_len) in dataloader:
        # import ipdb; ipdb.set_trace()
        x = x.to(device)
        
        pred_y = model( (x,seq_len) )                # class vector
        pred_class = torch.argmax( pred_y ,1 )          # class id
        pred_class = dataloader.dataset.batch_idx2label( pred_class ) # class string
        result_list += list( zip(test_id, pred_class ) )
    write_result( result_list )

def get_testloader(batch_size):
    with open("../cache/intent/vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = "../cache/intent/intent2idx.json"
    with open( intent_idx_path ) as f:
        intent2idx: Dict[str, int] = json.load(f)

    with open( "../data/intent/test.json" ) as f:
        data = json.load(f)
    test_dataset = SeqClsDataset( data, vocab, intent2idx, batch_size , mode="test" )

    # # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader( test_dataset,
            batch_size= batch_size,
            shuffle=False )
            # collate_fn= test_dataset.collate_fn_test )
    return test_loader

def main(args):
    device = args.device

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    test_dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len,mode="test")
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader( test_dataset,
            batch_size=args.batch_size,
            shuffle=False )
            # collate_fn= test_dataset.collate_fn_test )
    

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        test_dataset.num_classes,
    )
    # model.eval()
    # ckpt = torch.load(args.ckpt_path)

    # load weights into model
    load_model(model, args.ckpt_path )

    # TODO: predict dataset

    result_list = []
    for test_id, (x,seq_len) in tqdm(dataloader):
        # import ipdb; ipdb.set_trace()
        pred_y = model( (x,seq_len) )                # class vector
        pred_class = torch.argmax( pred_y ,1 )          # class id

        pred_class = test_dataset.batch_idx2label( pred_class ) # class string
        
        result_list += list( zip(test_id, pred_class ) )

    write_result( result_list )


import csv
from pathlib import Path
from model import get_now_time_str
def write_result(result_list):
    result_dir = "../"
    Path( result_dir ).mkdir( parents=True, exist_ok=True)
    result_path = result_dir +  'intent_' + get_now_time_str() + '.csv' 
    result_path = result_dir /  args.pred_file 
    with open( result_path ,'w',newline='') as f:
        w = csv.writer( f )
        w.writerow(["id","intent" ])               
        w.writerows( result_list )
    print( "write to " , result_path , " completed" )

        # TODO: write prediction to file (args.pred_file)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default = "../data/intent/test.json"
        # required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="../cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
