from typing import List, Dict
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence
from utils import Vocab


def get_datasets():
    import pickle , json
    SPLITS = [ "train", "eval"]

    cache_dir = "../cache/intent/"
    data_dir = "../data/intent/"
    max_len = 128
    with open( cache_dir + "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    # vocabulary, for label mapping
    intent_idx_path = cache_dir + "intent2idx.json"
    intent2idx = json.load( open(intent_idx_path,"r") )
    data_paths = {split: data_dir + f"{split}.json" for split in SPLITS}
    data = {split: json.load( open(path,'r') ) for split, path in data_paths.items()}

    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, max_len)
        for split, split_data in data.items()
    }
    test_data = json.load( open( data_dir + "test.json") )
    test_dataset = SeqClsDataset( test_data, vocab, intent2idx, max_len,"test" )
    return datasets , test_dataset



class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode = "train"
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = { idx: intent for intent, idx in self.label_mapping.items() }
        self.max_len = max_len
        self.mode = mode
        self._initial()

    def _initial(self):
        # data text --> encode,padding --> tensor
        sequences = [ instance['text'].split() for instance in self.data ]
        self.seq_len = [ len(seq) for seq in sequences ] 
        self.sequences = self.vocab.encode_batch( sequences ,self.max_len )
        self.sequences = torch.tensor( self.sequences )

        # train data: label2idx, to tensor
        if self.mode == "train":
            labels = [ self.label2idx( instance['intent'] ) for instance in self.data ]
            self.labels = torch.tensor( labels )


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        if self.mode == "train":
            return ( self.sequences[index], self.seq_len[index]), self.labels[index]
        else :
            return self.data[index]['id'],(self.sequences[index],self.seq_len[index])


        # instance = self.data[index]
        # seq_idx = self.vocab.encode( instance['text'] ) 
        # seq_idx = torch.tensor(seq_idx)

        # if self.mode == "train":
        #     label = self.label2idx( instance['intent'] )
        #     label = torch.tensor( label )
        #     return self.sequences[index], label
        # elif self.mode=="test":
        #     return  instance['id'] , seq_idx
        return 1


    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, batch ):
        # 由長到短排序
        batch.sort( key=lambda x:len(x[0]), reverse=True )
        # 取出 training data, label
        x,y = zip(*batch)
        # 紀錄原始 sequence 長度
        seq_len = [ len(seq) for seq in x ]
        x = pad_sequence( x ,batch_first=True )
        y = torch.stack(y)
        # x = pack_sequence(x)
        return (x,seq_len), y

    def collate_fn_test(self,batch):
        # 由長到短排序
        batch.sort( key=lambda x:len(x[1]), reverse=True )
        # 取出 training data, label
        test_id, x = zip(*batch)
        # 紀錄原始 sequence 長度
        # import ipdb; ipdb.set_trace()
        seq_len = [ len(seq) for seq in x ]
        x = pad_sequence( x ,batch_first=True )
        # test_id = torch.stack(test_id)
        return test_id, (x,seq_len)
        pass

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
    def batch_idx2label(self,batch):
        cls_num = batch.tolist()  # tensor --> int list
        classes = [  self.idx2label(num) for num in cls_num ]
        return classes


###
if __name__ == "__main__":

    datasets ,test_dataset = get_datasets()
    dataset = datasets['train']
    print( datasets )
    print( datasets['train'] )
    print( datasets['train'].data[0])
    ds =  datasets['train']
    ds[0]
###
    print( datasets['train'][0])
    print( datasets['train'][1])
    print( "\n\n" )

    print( datasets['eval'].data[0] )
    print( datasets['eval'][0] )

    print( "\n\n" )
    print( test_dataset )
    print( test_dataset.data[0] )
    print( test_dataset[0] )

    from torch.utils.data import Dataset, DataLoader

    dataoloader = DataLoader( dataset ,64 ,collate_fn = dataset.collate_fn, shuffle=True)

    # a =  next(iter(dataoloader)) 
    # import ipdb; ipdb.set_trace()
    for (x,seq_len),y in dataoloader:
        # import ipdb; ipdb.set_trace()
        print( x )
        print( seq_len )
        print( y )

    # print( a[0][27])



###



