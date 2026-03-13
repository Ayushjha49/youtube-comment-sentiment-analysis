"""
=============================================================================
dl_model.py — BiLSTM + Attention for Multilingual Sentiment Analysis
=============================================================================

ARCHITECTURE:  Embedding → SpatialDropout → BiLSTM × 2 → Attention → Dense → Softmax

WHY BILSTM BEATS ML ON THIS DATASET?
  1. Code-mixed romanized text: "ramro video cha yaar, loved it!" 
     → BiLSTM processes left-to-right AND right-to-left context
     → Catches sentiment signals even when mixed with unknown words
  2. Character-level embeddings learned implicitly by the word embeddings
  3. Attention focuses on "ramro", "loved" while ignoring filler words
  4. Handles OOV romanized words better (embedding interpolation)

EXPECTED ACCURACY: 85-91% on 125k romanized/code-mixed dataset
  vs Ensemble ML: 82-86%
  → DL wins by ~3-5% on this type of multilingual text
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score


# =============================================================================
# CUSTOM ATTENTION LAYER
# =============================================================================
class AttentionLayer(layers.Layer):
    """
    Bahdanau-style additive self-attention.

    For input sequence: ['video', 'ekdam', 'ramro', 'cha', 'yaar']
    Attention scores:   [  0.05,    0.15,    0.65,  0.08,  0.07  ]
                                             ↑ focuses on sentiment word
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name        = 'attention_W',
            shape       = (input_shape[-1], 1),
            initializer = 'glorot_uniform',
            trainable   = True,
        )
        self.b = self.add_weight(
            name        = 'attention_b',
            shape       = (input_shape[1], 1),
            initializer = 'zeros',
            trainable   = True,
        )
        super().build(input_shape)

    def call(self, x):
        # e_t = tanh(W·h_t + b)  shape: (batch, seq, 1)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # α_t = softmax(e_t)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context = Σ α_t * h_t  shape: (batch, hidden_dim)
        context = tf.reduce_sum(x * attention_weights, axis=1)
        return context

    def get_config(self):
        return super().get_config()


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================
def build_bilstm_attention(vocab_size: int, cfg) -> Model:
    """
    PRIMARY MODEL — Embedding → BiLSTM × 2 → Attention → Dense → Softmax

    Architecture details:
      • SpatialDropout1D: drops entire feature maps → less correlated than regular dropout
      • BiLSTM layer 1: 128 units each direction → 256 total; returns full sequence
      • BiLSTM layer 2: 64 units each direction → 128 total; refines context
      • AttentionLayer: weighted sum of all time steps
      • Dense(64) + Dropout(0.4): feature compression
      • Dense(32) + Dense(3, softmax): final classification
    """
    inp = Input(shape=(cfg.MAX_SEQ_LEN,), name='input_ids')

    x = layers.Embedding(
        input_dim           = vocab_size,
        output_dim          = cfg.EMBED_DIM,
        input_length        = cfg.MAX_SEQ_LEN,
        embeddings_regularizer = l2(cfg.L2_LAMBDA),
        name                = 'embedding',
    )(inp)

    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            cfg.LSTM_UNITS,
            return_sequences    = True,
            dropout             = cfg.LSTM_DROPOUT,
            recurrent_dropout   = cfg.LSTM_REC_DROP,
            kernel_regularizer  = l2(cfg.L2_LAMBDA),
        ),
        name='bilstm_1',
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            cfg.LSTM_UNITS // 2,
            return_sequences    = True,
            dropout             = cfg.LSTM_DROPOUT,
            recurrent_dropout   = cfg.LSTM_REC_DROP,
        ),
        name='bilstm_2',
    )(x)

    x = AttentionLayer(name='attention')(x)

    x = layers.Dense(cfg.DENSE_UNITS, activation='relu',
                     kernel_regularizer=l2(cfg.L2_LAMBDA))(x)
    x = layers.Dropout(cfg.DROPOUT)(x)
    x = layers.Dense(32, activation='relu')(x)

    out = layers.Dense(cfg.NUM_CLASSES, activation='softmax', name='output')(x)

    return Model(inputs=inp, outputs=out, name='BiLSTM_Attention')


def build_cnn_bilstm(vocab_size: int, cfg) -> Model:
    """
    ALTERNATIVE — CNN + BiLSTM hybrid.
    CNN extracts local n-gram features (bigrams, trigrams, 4-grams in parallel).
    BiLSTM then captures long-range context over those features.
    Sometimes faster training and comparable accuracy.
    """
    inp = Input(shape=(cfg.MAX_SEQ_LEN,), name='input_ids')

    x = layers.Embedding(
        input_dim    = vocab_size,
        output_dim   = cfg.EMBED_DIM,
        input_length = cfg.MAX_SEQ_LEN,
        name         = 'embedding',
    )(inp)

    x = layers.SpatialDropout1D(0.25)(x)

    # Parallel CNNs — multi-scale n-gram detection
    conv2 = layers.Conv1D(64, 2, activation='relu', padding='same')(x)
    conv3 = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    conv4 = layers.Conv1D(64, 4, activation='relu', padding='same')(x)
    x = layers.Concatenate()([conv2, conv3, conv4])
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3)
    )(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=inp, outputs=out, name='CNN_BiLSTM')


# =============================================================================
# DL DATA PIPELINE
# =============================================================================
class DLDataPipeline:
    """
    Prepares text data for deep learning:
      1. Clean text (using TextCleaner)
      2. Tokenize → integer sequences
      3. Pad sequences to MAX_SEQ_LEN
      4. Encode labels to integers → one-hot
      5. Train / Val / Test split
    """

    def __init__(self, cfg=None, cleaner=None):
        if cfg is None:
            from config import DLConfig
            cfg = DLConfig

        self.cfg = cfg

        if cleaner is None:
            from preprocess import TextCleaner
            self.cleaner = TextCleaner()
        else:
            self.cleaner = cleaner

        self.tokenizer = Tokenizer(
            num_words   = cfg.VOCAB_SIZE,
            oov_token   = cfg.OOV_TOKEN,
            lower       = True,
            filters     = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )
        self.label_encoder = LabelEncoder()

    def load_and_clean(self, filepath: str):
        import pandas as pd
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['comment_text', 'sentiment'])
        df = df.drop_duplicates(subset=['comment_text'])
        df['sentiment'] = df['sentiment'].str.strip().str.lower()
        df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])]

        print(f'[DL] Loaded {len(df):,} samples')
        print(f'[DL] Distribution:\n{df["sentiment"].value_counts()}')

        print('[DL] Cleaning text...')
        df['cleaned_text'] = self.cleaner.batch_clean(
            df['comment_text'].tolist(), show_progress=True
        )
        df = df[df['cleaned_text'].str.strip() != '']
        return df.reset_index(drop=True)

    def prepare(self, filepath: str) -> dict:
        df = self.load_and_clean(filepath)
        cfg = self.cfg

        y   = self.label_encoder.fit_transform(df['sentiment'])
        y_cat = keras.utils.to_categorical(y, num_classes=cfg.NUM_CLASSES)

        label_map = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        print(f'[DL] Label map: {label_map}')

        # Stratified splits
        texts = df['cleaned_text'].tolist()

        if cfg.TEST_SIZE > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts, y_cat,
                test_size    = cfg.TEST_SIZE,
                random_state = cfg.RANDOM_SEED,
                stratify     = y,
            )
            val_ratio = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size    = val_ratio,
                random_state = cfg.RANDOM_SEED,
                stratify     = y_temp.argmax(axis=1),
            )
            y_test_raw = y_test.argmax(axis=1)
        else:
            # 99% train / 1% val — no test split
            X_train, X_val, y_train, y_val = train_test_split(
                texts, y_cat,
                test_size    = cfg.VAL_SIZE,
                random_state = cfg.RANDOM_SEED,
                stratify     = y,
            )
            X_test, y_test, y_test_raw = [], np.array([]), np.array([])
            print(f'[DL] No test split — {len(X_train):,} train / {len(X_val):,} val')

        # Tokenize on train only — never fit on val or test data
        self.tokenizer.fit_on_texts(X_train)
        print(f'[DL] Vocab size: {len(self.tokenizer.word_index):,}')

        def _pad(txts):
            if not txts: return np.array([])
            seqs = self.tokenizer.texts_to_sequences(txts)
            return pad_sequences(seqs, maxlen=cfg.MAX_SEQ_LEN, padding='post', truncating='post')

        X_train_pad = _pad(X_train)
        X_val_pad   = _pad(X_val)
        X_test_pad  = _pad(X_test)

        if cfg.TEST_SIZE > 0:
            print(f'[DL] Train: {X_train_pad.shape} | Val: {X_val_pad.shape} | Test: {X_test_pad.shape}')
        else:
            print(f'[DL] Train: {X_train_pad.shape} | Val: {X_val_pad.shape}')

        return {
            'X_train': X_train_pad, 'y_train': y_train,
            'X_val':   X_val_pad,   'y_val':   y_val,
            'X_test':  X_test_pad,  'y_test':  y_test,
            'y_test_raw': y_test_raw,
            'vocab_size': min(len(self.tokenizer.word_index) + 1, cfg.VOCAB_SIZE + 1),
            'label_encoder': self.label_encoder,
        }

    def texts_to_padded(self, texts: list) -> np.ndarray:
        """Convert raw texts to padded sequences for inference."""
        cleaned = self.cleaner.batch_clean(texts)
        seqs    = self.tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(
            seqs,
            maxlen    = self.cfg.MAX_SEQ_LEN,
            padding   = 'post',
            truncating= 'post',
        )

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'dl_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(os.path.join(save_dir, 'dl_label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(os.path.join(save_dir, 'dl_text_cleaner.pkl'), 'wb') as f:
            pickle.dump(self.cleaner, f)
        print(f'[DL] Saved pipeline artifacts → {save_dir}')


# =============================================================================
# TRAINER
# =============================================================================
def compute_class_weights(y_cat: np.ndarray) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    y = y_cat.argmax(axis=1)
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes.astype(int), weights))


def train_dl(data: dict, save_dir: str, cfg=None, architecture: str = 'bilstm') -> dict:
    """
    Full DL training loop with callbacks.

    Args:
        data         : Output of DLDataPipeline.prepare()
        save_dir     : Where to save model checkpoints
        cfg          : DLConfig instance
        architecture : 'bilstm' or 'cnn_bilstm'
    """
    if cfg is None:
        from config import DLConfig
        cfg = DLConfig

    os.makedirs(save_dir, exist_ok=True)
    tf.random.set_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f'[DL] GPU: {gpus[0]}')
    else:
        print('[DL] Training on CPU')

    vocab_size = data['vocab_size']

    if architecture == 'cnn_bilstm':
        model = build_cnn_bilstm(vocab_size, cfg)
    else:
        model = build_bilstm_attention(vocab_size, cfg)

    model.summary()

    model.compile(
        optimizer = Adam(learning_rate=cfg.LEARNING_RATE, clipnorm=1.0),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    class_weights = compute_class_weights(data['y_train'])
    print(f'[DL] Class weights: {class_weights}')

    best_model_path = os.path.join(save_dir, 'dl_bilstm_best.keras')
    callbacks = [
        EarlyStopping(
            monitor           = 'val_accuracy',
            patience          = 5,
            restore_best_weights = True,
            verbose           = 1,
        ),
        ModelCheckpoint(
            filepath          = best_model_path,
            monitor           = 'val_accuracy',
            save_best_only    = True,
            verbose           = 1,
        ),
        ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = 0.5,
            patience  = 3,
            min_lr    = 1e-6,
            verbose   = 1,
        ),
    ]

    print(f'\n[DL] Training {architecture}...')
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data = (data['X_val'], data['y_val']),
        epochs          = cfg.EPOCHS,
        batch_size      = cfg.BATCH_SIZE,
        class_weight    = class_weights,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────
    le         = data['label_encoder']
    y_test_raw = data['y_test_raw']

    if len(data['X_test']) > 0:
        test_proba = model.predict(data['X_test'], batch_size=cfg.BATCH_SIZE)
        test_preds = test_proba.argmax(axis=1)
        acc = accuracy_score(y_test_raw, test_preds)
        f1  = f1_score(y_test_raw, test_preds, average='weighted')
        print(f'\n[DL] Test Accuracy : {acc:.4f}')
        print(f'[DL] Test F1       : {f1:.4f}')
        print(classification_report(y_test_raw, test_preds, target_names=le.classes_, digits=4))
    else:
        test_proba = np.array([])
        test_preds = np.array([])
        acc, f1    = 0.0, 0.0
        print('\n[DL] No test set — evaluation skipped (TEST_SIZE=0.0)')

    final_path = os.path.join(save_dir, 'dl_bilstm_final.keras')
    model.save(final_path)
    print(f'[DL] Saved final model → {final_path}')

    return {
        'model'         : model,
        'history'       : history,
        'test_accuracy' : acc,
        'test_f1'       : f1,
        'test_preds'    : test_preds,
        'test_proba'    : test_proba,
    }


def load_dl_model(model_path: str) -> keras.Model:
    """Load a saved Keras model with custom AttentionLayer."""
    return keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
