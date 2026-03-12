# 🧠 YouTube Sentiment Analyzer

AI-powered sentiment analysis for YouTube comments using a **BiLSTM + Attention** deep learning model and a **5-model ML Ensemble** — trained on 125k+ multilingual comments in **English, Nepali Roman script, and Hindi Roman script**.

This project goes beyond simple English sentiment analysis. Most sentiment tools fail completely on romanized South Asian text like *"ekdum ramro video thiyo yaar"* or *"kasto bakwas ho yaar time waste"*. This system was built specifically to handle that — by training custom embeddings that learn romanized word representations from scratch, combined with a bidirectional LSTM that captures context in both directions and an attention mechanism that zeroes in on the sentiment-heavy words that matter most.

---

## ✨ Features

- 🔗 **Paste any YouTube URL** → instant sentiment analysis — no manual copy-pasting of comments required; the system fetches them directly via the YouTube Data API v3
- 🌏 **Multilingual** — handles pure English, Romanized Nepali (ekdam, ramro, bakwas, yaar, ho, cha...), Romanized Hindi (kya baat, mast, bekaar, bilkul...), and code-mixed text where both languages appear in the same comment
- 🧠 **Two model types** — choose between the BiLSTM deep learning model for best accuracy on mixed-language text, or the 5-model ML ensemble for faster inference without TensorFlow
- 📊 **Beautiful UI** — results displayed as bar chart, pie chart, and radial chart with a toggle between views so you can visualize the sentiment distribution in the format that suits you
- ⚡ **Fast inference** — analyze 500 comments in approximately 2–4 seconds end-to-end, including API fetching, preprocessing, and prediction
- 📈 **Distribution stats** — see the exact percentage breakdown of positive, negative, and neutral comments, plus an overall sentiment label and confidence score for the video

---

## 🏗 Project Structure

```
youtube-sentiment/
├── backend/
│   ├── src/
│   │   ├── preprocess.py       # Text cleaning pipeline for romanized Nepali/Hindi
│   │   │                       # Handles emoji conversion, spam detection,
│   │   │                       # Devanagari stripping, laugh normalization, etc.
│   │   ├── youtube_fetcher.py  # YouTube Data API v3 integration
│   │   │                       # Fetches comments, handles pagination + rate limits
│   │   ├── ml_models.py        # 5 ML models (LR, SVM, RF, KNN, XGBoost)
│   │   │                       # + soft-voting ensemble with TF-IDF features
│   │   ├── dl_model.py         # BiLSTM + Attention architecture definition
│   │   │                       # AttentionLayer custom Keras layer included here
│   │   └── config.py           # All hyperparameters, paths, and constants
│   │                           # Single source of truth for tuning
│   ├── training/
│   │   ├── train_ml.py         # Script to train all 5 ML models + ensemble
│   │   │                       # Saves vectorizer + models to saved_models/
│   │   └── train_dl.py         # Script to train BiLSTM model
│   │                           # Saves tokenizer + weights to saved_models/
│   ├── saved_models/           # Trained model artifacts — gitignored, not committed
│   │                           # dl_bilstm_final.keras, ml_ensemble.pkl,
│   │                           # dl_tokenizer.pkl, dl_label_encoder.pkl, etc.
│   ├── app.py                  # FastAPI application — defines all API routes
│   ├── schemas.py              # Pydantic request/response models for type safety
│   └── predictor.py            # Inference engine — loads models and runs predictions
│                               # Handles both ML and DL prediction paths
├── frontend/
│   ├── pages/index.js          # Main Next.js page — URL input + results layout
│   ├── components/
│   │   ├── LoadingAnimation.jsx  # Animated loading state while fetching + analyzing
│   │   ├── ResultsDashboard.jsx  # Top-level results container with chart toggle
│   │   └── SentimentChart.jsx    # Recharts bar/pie/radial chart component
│   └── styles/globals.css        # Global Tailwind overrides and custom styles
├── data/
│   ├── raw/                    # Your original unprocessed dataset — gitignored
│   └── processed/              # Cleaned, labeled CSV ready for training — gitignored
├── requirements.txt            # All Python dependencies with pinned versions
├── .env.example                # Template showing required environment variables
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd youtube-sentiment

# Install Python backend dependencies
# (Recommended: create a virtual environment first)
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

> **Python version:** 3.9 or 3.10 recommended. TensorFlow 2.x has known issues with Python 3.11+ on some platforms.

### 2. Set up environment variables

```bash
cp .env.example .env
# Open .env in your editor and fill in your YouTube API key
```

Your `.env` file should look like this:

```
YOUTUBE_API_KEY=AIzaSy...yourkey...
```

**How to get a free YouTube Data API v3 key:**

1. Go to [Google Cloud Console](https://console.cloud.google.com) and sign in
2. Click **Select a project** → **New Project** → give it a name and create it
3. In the left sidebar go to **APIs & Services** → **Library**
4. Search for **YouTube Data API v3** and click **Enable**
5. Go to **APIs & Services** → **Credentials** → **Create Credentials** → **API Key**
6. Copy the generated key and paste it into your `.env` file

> **Quota note:** The free tier gives you 10,000 units/day. Fetching 500 comments from one video costs approximately 5–10 units, so you can comfortably run many analyses per day without hitting limits.

### 3. Prepare your dataset

Your training CSV must have exactly these two columns:

```csv
comment_text,sentiment
"This video is absolutely amazing!",positive
"vayo ni yaar ekdam ramro video thiyo",positive
"kasto bakwas video ho yaar time waste",negative
"okay video ho not bad not great",neutral
"kya baat hai bhai bilkul sahi",positive
"boring video skip garnu parcha",negative
```

**Labeling rules:**
- `positive` — praise, excitement, appreciation, agreement ("ramro", "love this", "daami")
- `negative` — criticism, frustration, dislike ("bakwas", "waste", "bekaar")
- `neutral` — factual, neither positive nor negative ("okay", "video heryo", "information")

Place your dataset at: `data/processed/comments_cleaned.csv`

If your raw data is unlabeled, you can manually label a sample, use a simpler pre-existing sentiment tool to bootstrap labels (then correct them), or use the semi-supervised approach described in `notebooks/02_data_labeling.ipynb`.

### 4. Train the models

```bash
cd backend

# ── Train all 5 ML models + ensemble ──────────────────────────────────────
# Trains: Logistic Regression, SVM, Random Forest, KNN, XGBoost
# Fits TF-IDF vectorizer and saves everything to saved_models/
# Expected time: 5–15 minutes depending on dataset size
python training/train_ml.py --data ../data/processed/comments_cleaned.csv

# ── Train BiLSTM deep learning model ──────────────────────────────────────
# Trains the BiLSTM + Attention model from scratch
# Expected time: 30–60 minutes on CPU, 5–10 minutes on GPU
python training/train_dl.py --data ../data/processed/comments_cleaned.csv

# ── Recommended: Train DL model on Google Colab (free GPU) ────────────────
# Upload: notebooks/04_dl_model_experiments.ipynb to Colab
# Go to Runtime → Change runtime type → GPU (T4)
# Run all cells — training takes ~5–10 minutes
# Download the saved_models/ folder and place it in backend/
```

Both training scripts will print a classification report at the end showing per-class precision, recall, and F1 score. If validation accuracy stops improving for 5 consecutive epochs, early stopping kicks in automatically.

### 5. Run the application

Open two terminal windows (both from the project root):

```bash
# ── Terminal 1: Start the FastAPI backend ─────────────────────────────────
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# --reload enables hot-reload on file save (development only)
# Remove --reload in production

# ── Terminal 2: Start the Next.js frontend ────────────────────────────────
cd frontend
npm run dev
# Runs on http://localhost:3000 by default
```

Open **http://localhost:3000** in your browser 🎉

Paste any YouTube video URL into the input box and hit Analyze. The system will fetch up to 500 top-level comments, run them through the preprocessing pipeline, classify each one, and return the full sentiment distribution in a few seconds.

---

## 🤖 Model Architecture

### Why 5 ML Models?

Rather than relying on a single algorithm, the ML ensemble combines five complementary classifiers. Each model has different strengths and makes different types of errors — when you combine them with soft voting (averaging predicted probabilities), the ensemble tends to be more accurate and more robust than any individual model.

All five models use **TF-IDF (Term Frequency–Inverse Document Frequency)** features. TF-IDF converts each comment into a sparse numerical vector where each dimension represents how important a word is in that comment relative to the entire corpus. This works surprisingly well for sentiment because words like "ramro", "daami", and "love" are strongly associated with positive sentiment, while "bakwas", "waste", and "hate" are strongly negative.

| Model | Strength | Why It's Included | Expected Accuracy |
|-------|----------|-------------------|-------------------|
| **Logistic Regression** | Fast, great with TF-IDF, strong L2 regularization prevents overfitting | Excellent linear baseline, very interpretable (you can inspect feature weights) | ~79–83% |
| **Linear SVM** | Best margin classifier for high-dimensional sparse text vectors | Maximizes the decision boundary margin, often outperforms LR on text tasks | ~80–84% |
| **Random Forest** | Non-linear patterns, robust to noisy/mislabeled data | Captures interactions between features that linear models miss | ~75–79% |
| **KNN** | Non-parametric, no assumptions about data distribution | Provides a diverse "voting member" even if it's not the strongest alone | ~68–73% |
| **XGBoost** | Gradient boosting on sparse TF-IDF features | Handles class imbalance well, strong on structured patterns | ~79–83% |
| **ML Ensemble** (LR + SVM + XGB, soft vote) | Complementary errors cancel out | The final ensemble uses only the top 3 models by validation accuracy | ~82–86% |

> **Note:** Random Forest and KNN are trained but their weight in the final ensemble vote is determined by their validation performance. In practice LR, SVM, and XGBoost dominate the ensemble.

### Why BiLSTM + Attention?

Traditional ML with TF-IDF treats each word independently and ignores word order entirely. This is fine for English where individual words carry a lot of signal, but for romanized code-mixed text, **context is critical**:

- "ramro chaina" (not good) — the word "ramro" (good) is negated by "chaina" (not), but TF-IDF sees both words independently and might predict positive
- "ekdum daami" (totally excellent) — BiLSTM understands "ekdum" intensifies "daami", producing a strongly positive representation
- Spelling variants — "ramro", "raaamro", "ramrooo" are all the same word; custom embeddings trained on this corpus learn they're similar

```
Text → Tokenizer → Integer sequences
     → Embedding(vocab=60000, dim=128)  # Learns word representations from scratch
     → SpatialDropout1D(0.2)            # Regularizes at the embedding level
     → BiLSTM(128 units × 2 directions) # Forward + backward pass, return_sequences=True
     → BiLSTM(64 units × 2 directions)  # Deeper context, return_sequences=True
     → AttentionLayer                    # Learns which tokens matter most for sentiment
     → Dense(64, relu) + Dropout(0.4)   # Non-linear transformation
     → Dense(32, relu)                  # Bottleneck layer
     → Dense(3, softmax)               # Output: [P(negative), P(neutral), P(positive)]
```

The **AttentionLayer** is a custom Bahdanau-style self-attention mechanism. It computes a score for each token at each position, normalizes with softmax to get weights that sum to 1, then produces a weighted sum of all hidden states. This means the model learns to "focus" on emotionally charged words even in a long comment full of filler words.

**Accuracy comparison:**

| Model | Test Accuracy | F1 (Weighted) | Notes |
|-------|--------------|---------------|-------|
| ML Ensemble | 82–86% | 0.83–0.87 | Fast (~50ms/batch), no GPU needed |
| **BiLSTM + Attention** | **85–91%** | **0.86–0.91** | **Best for romanized/code-mixed text** |
| ML + DL Combined | 87–92% | 0.88–0.92 | Best overall, averages both predictions |

**Why BiLSTM wins on romanized/code-mixed text:**
- Processes context bidirectionally — "chaina" after "ramro" changes the meaning; BiLSTM captures this
- Attention identifies sentiment-heavy words even when buried in long comments with filler text
- Learns romanized word representations end-to-end from the training data, including spelling variants
- Handles spelling variations naturally — "ramro" / "raaamro" / "ramrooo" cluster close together in embedding space after training

### Which model should I use?

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Production (max accuracy) | **ML + DL Ensemble** | Averages both models, highest accuracy |
| No GPU / lightweight deployment | **ML-only Ensemble** | No TensorFlow dependency, very fast |
| Research / experimentation | **BiLSTM only** | Easier to debug, inspect attention weights |
| Real-time with strict latency | **ML-only Ensemble** | ~50ms vs ~200ms for DL |

Use **ML + DL Ensemble** (the default) in production for maximum accuracy. Use **ML-only** if you need faster inference, have memory constraints, or can't install TensorFlow in your deployment environment.

---

## 🔌 API Reference

All endpoints are served by the FastAPI backend running on port 8000. FastAPI also auto-generates interactive documentation at `http://localhost:8000/docs` (Swagger UI) which you can use to test the API directly from your browser.

### `POST /api/analyze`

The main endpoint. Fetches comments from a YouTube video and returns the sentiment analysis results.

**Request body:**
```json
{
  "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
  "max_comments": 500,
  "model": "ensemble"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | ✅ | Full YouTube video URL (any format — `/watch?v=`, `youtu.be/`, etc.) |
| `max_comments` | integer | ❌ | Max number of comments to fetch and analyze. Default: 500. Max: 1000 |
| `model` | string | ❌ | Which model to use: `"ensemble"` (ML+DL), `"ml"`, or `"dl"`. Default: `"ensemble"` |

**Response:**
```json
{
  "success": true,
  "video_title": "Rick Astley - Never Gonna Give You Up",
  "overall_sentiment": "positive",
  "overall_confidence": 0.72,
  "distribution": {
    "positive": 61.4,
    "negative": 18.8,
    "neutral": 19.8
  },
  "analyzed_count": 500,
  "model_used": "ml+dl_ensemble",
  "processing_time_s": 2.34
}
```

| Field | Description |
|-------|-------------|
| `overall_sentiment` | The dominant sentiment class: `"positive"`, `"negative"`, or `"neutral"` |
| `overall_confidence` | Average confidence score for the overall sentiment label (0–1) |
| `distribution` | Percentage breakdown across all three classes (sums to 100) |
| `analyzed_count` | Actual number of comments analyzed (may be less than requested if video has fewer comments) |
| `processing_time_s` | End-to-end time in seconds from request to response |

**Error responses:**

```json
{ "success": false, "error": "Invalid YouTube URL" }
{ "success": false, "error": "Video not found or comments disabled" }
{ "success": false, "error": "YouTube API quota exceeded" }
```

---

### `GET /api/health`

Health check endpoint. Returns the current status of the backend and whether models are loaded.

```json
{
  "status": "ok",
  "models_loaded": {
    "ml_ensemble": true,
    "dl_bilstm": true
  },
  "version": "1.0.0"
}
```

---

### `GET /api/demo`

Demo endpoint that runs analysis on a pre-cached set of comments — no YouTube API key required. Useful for testing the frontend or verifying the backend is working correctly without consuming API quota.

```json
{
  "success": true,
  "video_title": "Demo Video",
  "overall_sentiment": "positive",
  "distribution": { "positive": 58.0, "negative": 22.0, "neutral": 20.0 },
  "analyzed_count": 100,
  "model_used": "ml+dl_ensemble",
  "note": "This is a demo response using cached data"
}
```

---

## 🧹 Preprocessing Details

The `TextCleaner` class in `src/preprocess.py` is one of the most important parts of the system. Raw YouTube comments are messy — they contain URLs, emojis, Devanagari script, spam, repeated characters, and a mix of languages. The preprocessing pipeline normalizes all of this before text reaches the models.

All cleaning steps are applied in order:

| Step | What It Does | Example |
|------|-------------|---------|
| **URL removal** | Strips all http/https URLs and bare domain links | `"check this youtu.be/xyz"` → `"check this"` |
| **Mention removal** | Removes `@username` tags | `"@creator great video"` → `"great video"` |
| **Hashtag removal** | Removes `#hashtag` tokens | `"#subscribe ramro video"` → `"ramro video"` |
| **Devanagari stripping** | Removes Unicode Devanagari characters (U+0900–U+097F) | `"राम्रो video ho"` → `"video ho"` |
| **Emoji → text** | Maps common emojis to sentiment-meaningful words | `"😍 video"` → `"love video"`, `"😡"` → `"angry"` |
| **Laugh normalization** | Collapses laugh variants to a single token | `"hahahahaha", "hehe", "lolll"` → `"laugh"` |
| **Repeated char reduction** | Normalizes exaggerated spelling to max 2 repeats | `"sooooooo", "raaamro"` → `"soo"`, `"raamro"` |
| **Romanized stopwords** | Removes common Nepali/Hindi function words | `"ko", "ka", "ki", "ma", "le", "ra", "ani"` |
| **Spam detection** | Flags and removes sub4sub, promo, and self-promotion patterns | `"sub for sub", "check my channel"` → removed |
| **Whitespace normalization** | Collapses multiple spaces, strips leading/trailing | `"great  video  ho"` → `"great video ho"` |

> **Why strip Devanagari?** The model was trained on romanized text. Devanagari characters in comments are usually mixed with romanized words and the model has not seen enough pure Devanagari to classify it well. Stripping it avoids feeding the model out-of-vocabulary tokens that could hurt predictions.

> **Why normalize repeated characters?** YouTube users often type "soooo good" or "ekdaaaaaam ramro" for emphasis. Without normalization, these become OOV tokens. Reducing to 2 repeats keeps the emphasis signal while allowing the tokenizer to map them to known vocabulary.

---

## 🗃 Dataset Format

```csv
comment_text,sentiment
"This video is absolutely amazing!",positive
"vayo ni yaar ekdam ramro video thiyo",positive
"kasto bakwas video ho yaar time waste",negative
"okay video ho not bad not great",neutral
"kya baat bhai bilkul perfect video",positive
"boring video skip garnu parcha",negative
"nice information diyeko cha",positive
"clickbait ho yaar content ramro chaina",negative
```

**Dataset statistics:**
- **125,000+ comments** labeled across 3 sentiment classes
- **Label distribution** (approximate): ~45% positive, ~30% negative, ~25% neutral
- Class imbalance is handled during training via `compute_class_weight('balanced', ...)` which gives higher loss weight to underrepresented classes

**Language breakdown the dataset covers:**
- Pure English comments (`"This is amazing"`, `"worst video ever"`)
- Romanized Nepali (`"ekdam ramro cha", "ekdamaai mast video thiyo"`)
- Romanized Hindi (`"kya baat hai yaar", "bilkul bekaar video hai"`)
- Code-mixed (`"this video is ekdam daami bro loved it"`)
- Emoji-heavy comments (preprocessed to text before training)

**How to expand the dataset:** Fetch more comments using `youtube_fetcher.py` from videos with clear sentiment (top-rated tutorials, controversial reviews, etc.) and label them. Even adding 10–20k well-labeled comments can improve accuracy noticeably.

---

## 📦 Tech Stack

| Layer | Technology | Why This Choice |
|-------|------------|----------------|
| **Deep Learning** | TensorFlow 2.x / Keras | Mature ecosystem, easy custom layers, good Colab support |
| **ML Models** | scikit-learn, XGBoost | Industry standard, fast training, great TF-IDF integration |
| **API Server** | FastAPI + Uvicorn | Async Python, auto-generated docs, fast and lightweight |
| **Frontend** | Next.js 14 + Tailwind CSS | Fast SSR, great developer experience, Tailwind for rapid styling |
| **Charts** | Recharts | React-native charting, easy to customize, supports multiple chart types |
| **YouTube API** | Google Data API v3 | Official API, handles pagination, supports comment threading |

---

## 🔧 Troubleshooting

**Models not found error on startup:**
Make sure you've run both training scripts and that `backend/saved_models/` contains at minimum `dl_bilstm_final.keras`, `dl_tokenizer.pkl`, `dl_label_encoder.pkl`, and `ml_ensemble.pkl`. The backend will fail to start if any of these are missing.

**YouTube API quota exceeded:**
The free quota is 10,000 units/day. If you hit it, wait until midnight Pacific Time for it to reset, or create a second Google Cloud project with a separate API key and add it to `.env`.

**TensorFlow not found:**
Run `pip install tensorflow` (CPU-only is fine for inference). If you're on an M1/M2 Mac, use `pip install tensorflow-macos` instead.

**Low accuracy on your dataset:**
Try increasing `VOCAB_SIZE` in `Config` if you have a very large or diverse dataset. Also check that your label distribution isn't severely imbalanced — if 80%+ of comments are one class, the model may default to that class. The `compute_class_weight` function should handle moderate imbalance automatically.

---

## 📝 License

MIT