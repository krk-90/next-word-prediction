import torch
import math
import spacy
from train import train
from preprocess import main

class evaluation:
    @torch.no_grad()
    def evaluate_loss(self,model, data_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_tokens += x_batch.size(0)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def evaluate_accuracy(self,model, data_loader, device, topk=(1, 5)):
        model.eval()
        max_k = max(topk)
        total = 0
        correct = {k: 0 for k in topk}

        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)  
            _, top_preds = outputs.topk(max_k, dim=1) 

            for k in topk:
                match = (top_preds[:, :k] == y_batch.unsqueeze(1)).any(dim=1)
                correct[k] += match.sum().item()

            total += y_batch.size(0)

        return {k: correct[k] / total for k in topk}


    def generate_text(self,model, seed_text, vocab, idx2word, seq_length=5,
                    num_words=20, device = None, temperature=0.5):
        nlp = spacy.load("en_core_web_sm",disable=["parser", "ner"])
        model.eval()
        tokens = [t.text.lower() for t in nlp(seed_text) if not t.is_space]
        generated = tokens.copy()

        for _ in range(num_words):
            context = generated[-seq_length:]
            if len(context) < seq_length:
                context = ["<PAD>"] * (seq_length - len(context)) + context
            encoded = torch.tensor(
                [[vocab.get(w, vocab["<UNK>"]) for w in context]],
                dtype=torch.long, device=device
            )
            logits = model(encoded).squeeze(0) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(idx2word.get(next_idx, "<UNK>"))

        return " ".join(generated)



def test():
    train_loader,test_loader,vocab = main()
    model,criterion = train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate = evaluation()
    test_loss, test_ppl = evaluate.evaluate_loss(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_ppl:.2f}")
    acc = evaluate.evaluate_accuracy(model, test_loader, device, topk=(1, 5))
    for k, v in acc.items():
        print(f"Top-{k} accuracy: {v*100:.2f}%")

    idx2word = {idx: word for word, idx in vocab.items()}    
    seq = 5
    for seed in ["who are "]:
        print(f"Seed: {seed!r}")
        print(" ->", evaluate.generate_text(model, seed, vocab, idx2word, device=device,seq_length=seq, num_words=15))

if __name__ == "__main__":
    test()