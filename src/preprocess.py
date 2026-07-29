from pathlib import Path
import pickle
import torch
import spacy
from torch.nn.utils.rnn import pad_sequence

class preprocess:
    def __init__(self,vocab):
        self.vocab = vocab
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.pad_idx = self.vocab["<PAD>"]
        self.unk_idx = self.vocab["<UNK>"]

    def text_to_seq(self,text):
        doc = self.nlp(text)
        return[self.vocab.get(token.text.lower(),self.unk_idx) for token in doc if not token.is_space]

    def pad_batch(self,batch):
        tensors = [torch.tensor(seq,dtype=torch.long) for seq in batch]
        return pad_sequence(tensors,batch_first=True,padding_value=self.pad_idx)
      
def main():
    file_path = Path(__file__).resolve().parent.parent
    path = file_path/"Data"/"tokens.pkl"
    with open(path,"rb") as f:
        vocab = pickle.load(f)
    preprocessor = preprocess(vocab=vocab)  

    seq =input("enter text:")

    batch = preprocessor.text_to_seq(seq)

    padded_batch = preprocessor.pad_batch([batch])

    print("Token IDs:", batch)
    print("Padded tensor:", padded_batch)
    print("Shape:", padded_batch.shape)
if __name__ == "__main__":
    main()