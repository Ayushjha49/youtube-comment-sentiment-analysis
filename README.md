# YouTube Sentiment Analyzer

Sentiment analysis for YouTube comments — built to handle **English, Romanized Nepali, Romanized Hindi, and code-mixed text**. Trained on 125k+ comments using a BiLSTM + Attention model and a 5-model ML ensemble.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Next.js](https://img.shields.io/badge/Next.js-14-black)

---

## What it does

Paste a YouTube URL → fetches comments → classifies each one as positive, negative, or neutral → shows a breakdown with charts.

Works well on comments like *"ekdam ramro video thiyo yaar"* or *"kasto bakwas ho time waste"* which most sentiment tools fail on completely.

---

## Stack

- **Backend** — FastAPI + Python
- **ML models** — scikit-learn, XGBoost (TF-IDF features)
- **DL model** — BiLSTM + Attention (TensorFlow/Keras)
- **Frontend** — Next.js 14 + Tailwind CSS + Recharts
- **Data source** — YouTube Data API v3

---

## Project structure

```
├── backend/
│   ├── src/
│   │   ├── preprocess.py       # text cleaning for romanized Nepali/Hindi
│   │   ├── youtube_fetcher.py  # YouTube API integration
│   │   ├── ml_models.py        # 5 ML models + soft-voting ensemble
│   │   ├── dl_model.py         # BiLSTM + Attention model
│   │   └── config.py           # hyperparameters and paths
│   ├── training/
│   │   ├── train_ml.py
│   │   └── train_dl.py
│   ├── saved_ml_models/        # gitignored
│   ├── saved_dl_models/        # gitignored
│   ├── app.py
│   ├── predictor.py
│   └── schemas.py
├── frontend/
│   ├── pages/
│   ├── components/
│   └── styles/
├── data/                       # gitignored
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Clone and install**
```bash
git clone <your-repo>
cd youtube-sentiment

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cd frontend && npm install
```

**2. Environment**
```bash
cp .env.example backend/.env
# add your YOUTUBE_API_KEY to backend/.env
```

Get a free API key at [console.cloud.google.com](https://console.cloud.google.com) → enable YouTube Data API v3 → Credentials → API Key.

**3. Train models**
```bash
cd backend

# ML models (~5-15 min)
python training/train_ml.py --data ../data/processed/comments_cleaned.csv

# DL model (~5-10 min on GPU, use Colab if no GPU)
python training/train_dl.py --data ../data/processed/comments_cleaned.csv
```

**4. Run**
```bash
# terminal 1
cd backend && uvicorn app:app --port 8000 --reload

# terminal 2
cd frontend && npm run dev
```

Open http://localhost:3000

---

## Models

| Model | Test Accuracy |
|-------|--------------|
| ML Ensemble (LR + SVM + XGB) | ~82-86% |
| BiLSTM + Attention | ~85-91% |
| ML + DL combined (default) | ~87-92% |

The BiLSTM model handles romanized/code-mixed text significantly better than the ML models because it learns word representations from scratch and processes context bidirectionally. The combined ensemble is used by default.

---

## Dataset format

```csv
comment_text,sentiment
"This video is amazing!",positive
"kasto bakwas video ho yaar",negative
"okay ho not bad",neutral
```

125k+ comments across English, Romanized Nepali, Romanized Hindi, and code-mixed text.

---

## API

`POST /api/analyze`
```json
{
  "url": "https://youtube.com/watch?v=...",
  "max_comments": 500,
  "model": "ensemble"
}
```

`GET /api/health` — check if models are loaded

`GET /api/demo` — test without an API key

---

## License

MIT
