import pickle
import torch
import torch.nn as nn 
from pathlib import Path

class Hybrid_model(nn.Module):
    def __init__(self,tokens_size,embedded_dim = 64,hidden_dim = 128):
        super(Hybrid_model,self).__init__()

        self.embedding = nn.Embedding(tokens_size,embedded_dim)
        self.lstm = nn.LSTM(embedded_dim,hidden_dim,batch_first=True)
        self.gru = nn.GRU(embedded_dim,hidden_dim,batch_first=True)

        self.fc =nn.Sequential(
            nn.Linear(hidden_dim*2,64),
            nn.ReLU(),
            nn.Linear(64,tokens_size)
        )

    def forward(self,x):
        x = self.embedding(x)
        lstm_output,_ = self.lstm(x)    
        gru_output,_ = self.gru(x)
        lstm_last = lstm_output[:,-1,:]
        gru_last = gru_output[:,-1,:]
        combined = torch.cat((lstm_last,gru_last),dim=-1)
        return self.fc(combined)

def model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    file_path = Path(__file__).resolve().parents[2]
    tokens_path = file_path/"Data"/"tokens.pkl"
    params_path = file_path/"Data"/"params.pth"
    if not (tokens_path.exists() and params_path.exists()):
        raise FileNotFoundError(f"Missing {tokens_path} or {params_path}")

    with open(tokens_path,"rb") as f:
        tokens = pickle.load(f)

    parameters = torch.load(params_path,map_location=device,weights_only=True)    

    hybrid_moded = Hybrid_model(tokens_size=len(tokens))
    hybrid_moded.load_state_dict(parameters["model_state_dict"])
    hybrid_moded.to(device)
    hybrid_moded.eval()
    return hybrid_moded
if __name__ == "__main__":
    MODEL = model()