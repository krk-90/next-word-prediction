from pathlib import Path
import pickle
import torch
import spacy
from torch.nn.utils.rnn import pad_sequence

class preprocess:
    def __init__(self,vocab,seq_length = 5):
        self.vocab = vocab
        self.seq_length = seq_length
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

    def text_to_seq(self,text):
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_space]
        if len(tokens) < self.seq_length:
            raise ValueError(f"need at leasst {self.seq_length} words,input has {len(tokens)}")

        tokens = tokens[-self.seq_length:]
        return [self.vocab.get(tok,self.unk_idx) for tok in tokens]

    def to_tensor(self,seq):
        return torch.tensor([seq],dtype=torch.long)
    
def main():
    file_path = Path(__file__).resolve().parent.parent
    path = file_path/"Data"/"tokens.pkl"
    with open(path,"rb") as f:
        vocab = pickle.load(f)

    preprocessor = preprocess(vocab=vocab)  

    text =input("enter text:")

    seq = preprocessor.text_to_seq(text)
    x = preprocessor.to_tensor(seq)
    print("Token IDs:", seq)
    print("Tensor:", x)
    print("Shape:", x.shape)
    return x
if __name__ == "__main__":
    main()