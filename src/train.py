import torch
import torch.nn as nn
from pathlib import Path
from model import Hybrid_model
from preprocess import main

def train_model(model,train_loader,criterion,optimizer,device,epochs = 10,check_point_path = "checkpoint.pth"):
    model.to(device)
    start_epoch = 0
    try:
        checkpoint = torch.load(check_point_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        else:
            model.load_state_dict(checkpoint)
            print("Loaded weights only, starting fresh optimizer.")   
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"Could not load checkpoint ({e}), starting fresh.")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_train,y_train in train_loader:
            x_train,y_train = x_train.to(device),y_train.to(device)
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs,y_train)
            loss.backward()
            optimizer.step()

            total_train_loss +=loss.item()
        avg_train_loss = total_train_loss/len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, ")

        torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_train_loss,
}, check_point_path)

    return model

def train():
    train_loader,test_loader,vocab = main()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Hybrid_model(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    check_point_path = Path(__file__).resolve().parent / "Data" / "params.pth"
    check_point_path.parent.mkdir(parents=True, exist_ok=True)    
    trained_model = train_model(model=model,train_loader=train_loader,criterion=criterion,optimizer=optimizer,device=device,check_point_path=check_point_path)
    return trained_model,criterion
if __name__ == "__main__":
    trained_model = train()
