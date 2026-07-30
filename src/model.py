import torch
import torch.nn as nn


class Hybrid_model(nn.Module):
    def __init__(self, vocab_size,embedded_dim = 64,hidden_dim = 128):
        super(Hybrid_model,self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedded_dim)
        self.lstm = nn.LSTM(embedded_dim,hidden_dim,batch_first=True)
        self.gru = nn.GRU(embedded_dim,hidden_dim,batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2,64),
            nn.ReLU(),
            nn.Linear(64,vocab_size)
        )

    def forward(self,x):
        x = self.embedding(x) 
        lstm_out,_ = self.lstm(x)
        gru_out,_ = self.gru(x)   
        lstm_last = lstm_out[:,-1,:]
        gru_last = gru_out[:,-1,:]
        x = torch.cat((lstm_last, gru_last), dim=-1)
        x = self.fc(x)
        return x

    