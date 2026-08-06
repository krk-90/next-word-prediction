from pathlib import Path
import pickle
import torch
import re

class Preprocess:
    def __init__(self, vocab, seq_length=5):
        self.vocab = vocab
        self.seq_length = seq_length
        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

    def tokenize(self, text: str):
        text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text)

    def text_to_seq(self, text):
        tokens = self.tokenize(text)
        if len(tokens) < self.seq_length:
            raise ValueError(f"need at least {self.seq_length} words,input has {len(tokens)}")

        tokens = tokens[-self.seq_length:]
        return [self.vocab.get(tok, self.unk_idx) for tok in tokens]

    def to_tensor(self,seq):
        return torch.tensor([seq],dtype=torch.long)
    
def main(text:str = "Enter a sequence of text."):
    file_path = Path(__file__).resolve().parent.parent
    path = file_path/"Data"/"tokens.pkl"
    with open(path,"rb") as f:
        vocab = pickle.load(f)

    preprocessor = Preprocess(vocab=vocab)  
    seq = preprocessor.text_to_seq(text)
    x = preprocessor.to_tensor(seq)
    return x

if __name__ == "__main__":
    main()