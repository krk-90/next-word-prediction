import torch
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel,field_validator
from contextlib import asynccontextmanager
import pickle
from pathlib import Path

from  slowapi import Limiter,_rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .model import model
from .preprocess import Preprocess

MODEL = None
VOCAB = None
IDX2WORD = None

limiter = Limiter(key_func= get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VOCAB, IDX2WORD, PREPROCESSOR
    try:
        MODEL = model()
        file_path = Path(__file__).resolve().parent.parent
        path = file_path / "Data" / "tokens.pkl"
        with open(path, "rb") as f:
            VOCAB = pickle.load(f)
        IDX2WORD = {idx: word for word, idx in VOCAB.items()}
        PREPROCESSOR = Preprocess(vocab=VOCAB)  
        print("Model and vocab loaded successfully")
    except Exception as e:
        print(f"Error in loading model: {e}")
        raise
    yield

app = FastAPI(title="Next word prediction",version="1.0.0",lifespan=lifespan)   
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

class Input_text(BaseModel):
    text : str
    num_words: int = 1

    @field_validator("num_words")
    @classmethod
    def cap_num_words(cls,v):
        if v <1 or v>20:
            raise ValueError("num_words must be between 1 and 20")
        return v

class Output(BaseModel):
    predicted_text : str

@app.get("/")
def end_point():
    return {"message":"your Next word prediction model is running."}

@app.get("/status")
def status():
    return {
        "status":"unhealthy" if MODEL is None else "healthy" ,
        "model": MODEL is not None
    }

@app.post("/predict/",response_model=Output)
@limiter.limit("10/minute")
async def predict(request:Request,user_input:Input_text):
    if MODEL is None:
        raise HTTPException(status_code=500,detail="Model not loaded.")

    try:
        text = user_input.text
        generated_words = []

        seq = PREPROCESSOR.text_to_seq(text)
        for _ in range(user_input.num_words):
            x = PREPROCESSOR.to_tensor(seq)
            with torch.no_grad():
                output = MODEL(x)
                probs = torch.softmax(output, dim=-1)
                predict_idx = torch.multinomial(probs, num_samples=1).item()

            predict_word = IDX2WORD.get(predict_idx, "<UNK>")
            generated_words.append(predict_word)
            seq = seq[1:] + [predict_idx]

        return {"predicted_text": " ".join(generated_words)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
