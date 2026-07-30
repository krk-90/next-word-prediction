import torch
import spacy
import pickle
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader,TensorDataset

class preprocess:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm",disable=["parser", "ner"])

    def load_data(self):
        data_path = Path("/kaggle/input/datasets/vivekmettu/wikitext2-data")
        train_data_path = data_path + "/train.txt"
        test_data_path = data_path + "/test.txt"
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(test_data_path,"r",encoding="utf-8") as f:
            test_text = f.read()
        return train_text,test_text    
    
    def vocabulary(self,train_data,test_data):
        train_tokens = []
        test_tokens = []
        for doc in self.nlp.pipe(train_data.splitlines(), batch_size=1000):
            train_tokens.extend([token.text.lower() for token in doc if not token.is_space])

        for doc in self.nlp.pipe(test_data.splitlines(), batch_size=1000):
            test_tokens.extend([token.text.lower() for token in doc if not token.is_space])  

        vocab_counter = Counter(train_tokens)
        vocab = {"<PAD>":0,"<UNK>":1}
        vocab.update({word:idx+2 for idx,(word,_) in enumerate(vocab_counter.items())})

        return train_tokens,test_tokens,vocab

    def encode_tokens(self,tokens, vocab):
        return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    def create_sequences(self,encoded_tokens, seq_length):
        sequences = []
        labels = []
        for i in range(seq_length,len(encoded_tokens)):
            seq = encoded_tokens[i-seq_length:i]
            label = encoded_tokens[i]
            sequences.append(seq)
            labels.append(label)
        return sequences, labels

def main():
    preprocessor = preprocess()
    train_data,test_data = preprocessor.load_data()
    train_tokens, test_tokens, vocab = preprocessor.vocabulary(train_data,test_data)
    with open("tokens.pkl","wb") as f:
        pickle.dump(vocab,f)

    train_encoded = preprocessor.encode_tokens(train_tokens, vocab)
    test_encoded = preprocessor.encode_tokens(test_tokens, vocab)

    train_sequences, train_labels = preprocessor.create_sequences(train_encoded, seq_length=5)
    x_train,y_train = torch.tensor(train_sequences,dtype=torch.long),torch.tensor(train_labels,dtype=torch.long)
    train_dataset = TensorDataset(x_train,y_train)
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

    seq = 5
    test_seqs,test_labels = preprocessor.create_sequences(test_encoded,seq_length=seq)
    x_test,y_test = torch.tensor(test_seqs,dtype=torch.long),torch.tensor(test_labels,dtype=torch.long)
    test_dataset = TensorDataset(x_test,y_test)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)
    return train_loader,test_loader

if __name__ == "__main__":
    main()