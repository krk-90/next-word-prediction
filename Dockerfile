FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

RUN mkdir -p app/Data && \
    curl -sL -o app/Data/params.pth "https://media.githubusercontent.com/media/krk-90/next-word-prediction/main/app/Data/params.pth"

WORKDIR /app/app

CMD uvicorn backend.predict:app --host 0.0.0.0 --port ${PORT:-8000}