# Next Word Prediction

A hybrid **LSTM + GRU** deep learning model (PyTorch) for next-word prediction, trained on the WikiText-2 corpus and served through a **FastAPI** backend.

Given a sequence of text, the model predicts the most probable next word(s), combining the outputs of a parallel LSTM and GRU branch over a shared word embedding.

## How it works

The core model (`Hybrid_model`) embeds each input token, then feeds the embedding through an **LSTM** and a **GRU** in parallel. The final hidden states of both branches are concatenated and passed through a fully connected head to produce a probability distribution over the vocabulary.

```
tokens → Embedding (+Dropout) → ┌─ LSTM ─┐
                                 └─ GRU  ─┘ → concat(last hidden states) → Linear → ReLU → Dropout → Linear → vocab logits
```

- Sequence length used for training: 5 tokens of context predicting the 6th
- Tokenization/lowercasing handled by spaCy's `en_core_web_sm` (parser and NER disabled for speed)
- Special tokens: `<PAD>` and `<UNK>`

## Project structure

```
next-word-prediction/
├── model.ipynb              # End-to-end training/experimentation notebook (Kaggle, GPU)
├── dataset/
│   ├── train.txt             # WikiText-2 training split
│   └── test.txt               # WikiText-2 test split
├── src/                      # Standalone training pipeline (mirrors the notebook)
│   ├── preprocess.py         # Loads data, builds vocab, tokenizes, creates (context, label) sequences
│   ├── model.py               # Hybrid_model definition (LSTM + GRU)
│   ├── train.py                # Training loop with checkpointing/resuming
│   └── evaluate.py            # Loss/perplexity, top-k accuracy, and text generation
└── app/
    ├── Data/                  # Artifacts consumed by the API (train.txt, test.txt, tokens.pkl, params.pth)
    └── backend/
        ├── model.py            # Loads tokens.pkl + params.pth and rebuilds Hybrid_model for inference
        ├── preprocess.py       # Converts raw input text into a model-ready tensor
        └── predict.py          # FastAPI app exposing prediction endpoints
```

## Dataset

The model is trained on **WikiText-2**, a collection of Wikipedia articles commonly used for language modeling benchmarks. Text is lowercased and tokenized with spaCy, then converted into a fixed vocabulary (`<PAD>` = 0, `<UNK>` = 1, plus all training tokens).

## Getting started

### 1. Install dependencies

```bash
pip install torch spacy fastapi uvicorn pydantic
python -m spacy download en_core_web_sm
```

### 2. Train the model

The `src/` pipeline builds the vocabulary, encodes the data, trains the hybrid model, and checkpoints it to `app/Data/params.pth`:

```bash
cd src
python preprocess.py   # builds and saves tokens.pkl
python train.py        # trains the model, saves checkpoint (model_state_dict + optimizer_state_dict)
```

Training automatically resumes from an existing checkpoint if one is found at the target path.

### 3. Evaluate

```bash
cd src
python evaluate.py
```

This reports:
- Test loss and perplexity
- Top-1 / Top-5 accuracy
- Sample generated text from a seed phrase

### 4. Run the prediction API

The FastAPI backend in `app/backend` loads the saved vocabulary (`tokens.pkl`) and trained weights (`params.pth`) at startup and serves predictions:

```bash
cd app
uvicorn backend.predict:app --reload
```

#### Endpoints

| Method | Path        | Description                                   |
|--------|-------------|------------------------------------------------|
| GET    | `/`         | Health message confirming the API is running   |
| GET    | `/status`   | Reports whether the model loaded successfully   |
| POST   | `/predict/` | Predicts the next word(s) given input text      |

**Example request:**

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"text": "the history of the", "num_words": 3}'
```

**Example response:**

```json
{
  "predicted_text": "world war ii"
}
```

Predictions are sampled (via `torch.multinomial` over the softmax output) one word at a time, appending each generated word back into the context for the next step — so `num_words` controls how many words are generated in sequence.

> Note: the input text must contain at least as many tokens as the model's context window (5) once tokenized, or an error is raised.

## Model details

| Component        | Value                          |
|-------------------|--------------------------------|
| Embedding dim      | 64 (128 in the notebook version) |
| Hidden dim (LSTM/GRU) | 128 (256 in the notebook version) |
| Context window     | 5 tokens |
| Loss               | Cross-entropy |
| Optimizer          | Adam (lr = 0.001) |

## Notes

- `model.ipynb` reflects the original experimentation on Kaggle (GPU, larger embedding/hidden sizes, train/validation split, early stopping, LR scheduling). `src/` is a lighter-weight, script-based reproduction of the same pipeline.
- `app/Data` mirrors `dataset/` plus the serialized vocabulary (`tokens.pkl`) and trained weights (`params.pth`) needed to serve the API without retraining.