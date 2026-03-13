# # 🧠 BiLSTM + Attention — Maximum Accuracy Training
# 
# ### What's different from the original
# | | Original | This version |
# |---|---|---|
# | Train split | 70% | **99%** |
# | Val split | 15% | **1%** (early stopping only) |
# | Test split | 15% | **0%** (test on real YouTube comments) |
# | MAX_SEQ_LEN | 150 (hardcoded) | **auto from your data** (95th percentile) |
# | VOCAB_SIZE | 60k | **80k** (more OOV coverage) |
# | EMBED_DIM | 128 | **200** (richer multilingual representations) |
# | LR schedule | flat 1e-3 | **warmup + cosine decay** |
# | Label smoothing | none | **0.1** (prevents overconfidence) |
# | Early stopping patience | 5 | **7** |
# | Emoji map | 😭 = sad ❌ | **😭 = emotional** ✅ + 20 more |
# | Mixed precision | no | **yes (2x faster on T4)** |
# | Saves to Drive | no | **yes — safe from disconnects** |
# 
# ### ⚡ Before running
# **Runtime → Change runtime type → T4 GPU**

# ## 1. Install Dependencies

# ── Cell 3 ──────────────────────────────────────────────────
# Colab has TensorFlow pre-installed. Only uncomment if something is missing.
# !pip install -q scikit-learn pandas matplotlib seaborn tqdm

# ## 2. Mount Google Drive
# 
# **Your Drive folder must contain:**
# - `dl_model.py`
# - `config.py`
# - `preprocess.py`
# - `comments_cleaned.csv`
# 
# Models and checkpoints will save directly to Drive during training —
# safe even if Colab disconnects.

# ── Cell 5 ──────────────────────────────────────────────────

import os, shutil

# ── Set this to your actual Drive folder name ─────────────────────────────
DRIVE_FOLDER = 'yt_sentiment'   # change to whatever your folder is named
# ──────────────────────────────────────────────────────────────────────────

DRIVE_PATH   = f'/content/drive/MyDrive/{DRIVE_FOLDER}'
SAVE_DIR     = f'{DRIVE_PATH}/saved_dl_models'
DATA_PATH    = f'{DRIVE_PATH}/comments_cleaned.csv'

os.makedirs(SAVE_DIR, exist_ok=True)

# Copy source files from Drive to /content/ so imports work
for f in ['dl_model.py', 'config.py', 'preprocess.py']:
    src = f'{DRIVE_PATH}/{f}'
    dst = f'/content/{f}'
    if os.path.isfile(src):
        shutil.copy(src, dst)
        print(f'Copied {f}')
    else:
        print(f'ERROR: {f} not found in {DRIVE_PATH}')

print(f'\nDrive folder : {DRIVE_PATH}')
print(f'Save dir     : {SAVE_DIR}')
print(f'Data path    : {DATA_PATH}')
print(f'Data exists  : {os.path.isfile(DATA_PATH)}')

# ## 3. Imports

# ── Cell 7 ──────────────────────────────────────────────────
import os, sys, time, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

sys.path.insert(0, '/content')
from dl_model import DLDataPipeline, build_bilstm_attention, build_cnn_bilstm, compute_class_weights, load_dl_model
from config import DLConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

import tensorflow as tf
print(f'TensorFlow : {tf.__version__}')
print(f'GPU        : {len(tf.config.list_physical_devices("GPU")) > 0}')

# ## 4. GPU + Mixed Precision
# 
# Mixed precision (float16 compute, float32 weights) gives **~2x faster training** on T4 — completely free speedup.

# ── Cell 9 ──────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print(f'✅ GPU ready       : {gpus[0]}')
    print(f'✅ Mixed precision : float16 compute / float32 weights')
else:
    print('⚠️  No GPU — go to Runtime → Change runtime type → T4 GPU')

# ## 5. Configuration
# 
# Only change `ARCHITECTURE` here. `DATA_PATH` and `SAVE_DIR` were set in Cell 2.

# ── Cell 11 ──────────────────────────────────────────────────
ARCHITECTURE = 'bilstm'   # 'bilstm' (recommended) or 'cnn_bilstm'

# ── Splits: 99% train / 1% val / 0% test ─────────────────────────────────
# No test split — real testing happens on actual YouTube comments in production.
# Val set (1%) is kept ONLY for early stopping — without it the model overfits.
DLConfig.TEST_SIZE     = 0.0
DLConfig.VAL_SIZE      = 0.01

# ── Vocabulary: 80k covers far more OOV romanized Hindi/Nepali words ───────
DLConfig.VOCAB_SIZE    = 80000

# ── Embedding: 200-dim — standard for multilingual models (same as GloVe) ──
DLConfig.EMBED_DIM     = 200

# ── Training ──────────────────────────────────────────────────────────────
DLConfig.EPOCHS        = 50     # early stopping will find the real best
DLConfig.LEARNING_RATE = 3e-4   # warmup will ramp this up safely
DLConfig.BATCH_SIZE    = 512    # fills T4 memory efficiently

# ── MAX_SEQ_LEN is set AUTOMATICALLY in step 7 from your actual data ───────

print(f'SAVE_DIR  : {SAVE_DIR}')
print(f'DATA_PATH : {DATA_PATH}')
print('Configuration set. MAX_SEQ_LEN will be computed from your data in step 7.')

# ## 6. Emoji Map Fix
# 
# Key fixes for Hindi/Nepali YouTube culture:
# - **😭 → `emotional`** (was `sad`) — in South Asian YouTube comments 😭 means *overwhelmed/touched*, not sad
# - **💀 → `funny`** — modern usage: *I'm dead laughing*
# - **20+ emojis added** that were completely missing from the original map

# ── Cell 13 ──────────────────────────────────────────────────
def patch_emoji_map(cleaner):
    cleaner._emoji_map.update({
        # Cultural fixes for Hindi/Nepali YouTube
        '\U0001f62d': ' emotional ',   # 😭 was 'sad' — WRONG for this culture
        '\U0001f480': ' funny ',        # 💀 = 'I'm dead laughing'
        '\U0001f972': ' emotional ',    # 🥺 pleading/touched
        '\U0001f923': ' funny ',        # 🤣
        # Positive signals
        '\U00002728': ' amazing ',      # ✨
        '\U0001f31f': ' amazing ',      # 🌟
        '\U0001f4ab': ' amazing ',      # 💫
        '\U0001f64c': ' amazing ',      # 🙌
        '\U0001f44f': ' amazing ',      # 👏
        '\U0001f389': ' happy ',        # 🎉
        '\U0001f38a': ' happy ',        # 🎊
        '\U0001f970': ' love ',         # 🥰
        '\U0001f495': ' love ',         # 💕
        '\U0001f49e': ' love ',         # 💞
        '\U0001f48b': ' love ',         # 💋
        '\U0001faf6': ' amazing ',      # 🫶 heart hands
        '\U0001f9e1': ' love ', '\U0001f49b': ' love ',
        '\U0001f49a': ' love ', '\U0001f499': ' love ', '\U0001f49c': ' love ',
        # Negative signals
        '\U0001f615': ' disappointed ', # 😕
        '\U0001f614': ' sad ',          # 😔
        '\U0001f625': ' sad ',          # 😢
        '\U0001f630': ' scared ',       # 😰
        '\U0001f4a9': ' bad ',          # 💩
        # Neutral/mixed
        '\U0001f914': ' thinking ',     # 🤔
        '\U0001f60f': ' sarcastic ',    # 😏
        '\U0001f611': ' boring ',       # 😑
    })
    return cleaner

print('Emoji patch defined.')

# ## 7. Auto-Detect MAX_SEQ_LEN from Your Data
# 
# Setting MAX_SEQ_LEN too high wastes compute on zero-padding.  
# Setting it too low cuts off real signal from longer comments.  
# 
# **Best practice:** use the **95th percentile** of actual token lengths in your dataset.

# ── Cell 15 ──────────────────────────────────────────────────
from preprocess import TextCleaner
from tensorflow.keras.preprocessing.text import Tokenizer

print('Loading dataset to compute token length distribution...')
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.dropna(subset=['comment_text', 'sentiment'])
df_raw['sentiment'] = df_raw['sentiment'].str.strip().str.lower()
df_raw = df_raw[df_raw['sentiment'].isin(['positive', 'negative', 'neutral'])]
df_raw = df_raw.drop_duplicates(subset=['comment_text'])
print(f'Dataset size: {len(df_raw):,} samples')
print(f'Distribution:\n{df_raw["sentiment"].value_counts()}')

print('\nCleaning sample for length analysis (this takes ~30s)...')
_cleaner = patch_emoji_map(TextCleaner())
_sample  = df_raw['comment_text'].sample(min(20000, len(df_raw)), random_state=42).tolist()
_cleaned = _cleaner.batch_clean(_sample)
_cleaned = [t for t in _cleaned if t.strip()]

token_counts = [len(t.split()) for t in _cleaned]

p50  = int(np.percentile(token_counts, 50))
p90  = int(np.percentile(token_counts, 90))
p95  = int(np.percentile(token_counts, 95))
p99  = int(np.percentile(token_counts, 99))
pmax = int(np.max(token_counts))

print(f'\nToken length distribution (after cleaning):')
print(f'  p50  (median)     : {p50} tokens')
print(f'  p90               : {p90} tokens')
print(f'  p95               : {p95} tokens  ← we use this')
print(f'  p99               : {p99} tokens')
print(f'  max               : {pmax} tokens')

AUTO_SEQ_LEN = int(np.ceil(p95 / 16) * 16)
AUTO_SEQ_LEN = max(64, min(AUTO_SEQ_LEN, 256))
DLConfig.MAX_SEQ_LEN = AUTO_SEQ_LEN

print(f'\n✅ MAX_SEQ_LEN set to {AUTO_SEQ_LEN} (95th pct rounded to nearest 16)')
print(f'   This covers {sum(1 for c in token_counts if c <= AUTO_SEQ_LEN)/len(token_counts)*100:.1f}% of comments fully.')

plt.figure(figsize=(11, 4))
plt.hist([min(c, 250) for c in token_counts], bins=60, color='steelblue', alpha=0.8, edgecolor='white')
plt.axvline(AUTO_SEQ_LEN, color='red',    linestyle='--', linewidth=2, label=f'MAX_SEQ_LEN = {AUTO_SEQ_LEN} (p95)')
plt.axvline(p50,          color='orange', linestyle='--', linewidth=1.5, label=f'median = {p50}')
plt.title('Token Length Distribution (after cleaning)')
plt.xlabel('Token count'); plt.ylabel('Comments')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
del df_raw, _sample, _cleaned

# ## 8. Learning Rate Schedule
# 
# **Linear warmup (3 epochs)** → ramps LR from near-zero to peak, avoiding unstable early gradients.  
# **Cosine decay** → smoothly reduces LR to near-zero by final epoch.

# ── Cell 17 ──────────────────────────────────────────────────
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, peak_lr, warmup_epochs=3, total_epochs=50, min_lr=1e-6):
        super().__init__()
        self.peak_lr       = peak_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.peak_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.model.optimizer.learning_rate.assign(float(lr))
        if epoch < self.warmup_epochs or epoch % 5 == 0:
            print(f'  [LR] Epoch {epoch+1}: {lr:.2e}')

# Preview
_lrs = []
for e in range(50):
    if e < 3: _lrs.append(3e-4 * (e+1) / 3)
    else:
        p = (e-3)/47
        _lrs.append(1e-6 + 0.5*(3e-4-1e-6)*(1+np.cos(np.pi*p)))
plt.figure(figsize=(10, 3))
plt.plot(range(1, 51), _lrs, 'b-o', markersize=3)
plt.title('LR Schedule: 3-epoch warmup + cosine decay')
plt.xlabel('Epoch'); plt.ylabel('LR'); plt.yscale('log')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
print('LR schedule defined.')

# ## 9. Data Pipeline
# 
# **99% train / 1% val / 0% test.**  
# The val set is used only for early stopping — the model never trains on it.

# ── Cell 19 ──────────────────────────────────────────────────
import dl_model as _dlm
from sklearn.model_selection import train_test_split as _tts
from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad_seqs
from tensorflow.keras import utils as _ku

_orig_prepare = _dlm.DLDataPipeline.prepare

def patched_prepare(self, filepath):
    """Patched to support TEST_SIZE=0.0 (no test split)."""
    df = self.load_and_clean(filepath)
    cfg = self.cfg

    y     = self.label_encoder.fit_transform(df['sentiment'])
    y_cat = _ku.to_categorical(y, num_classes=cfg.NUM_CLASSES)

    label_map = dict(zip(self.label_encoder.classes_,
                         self.label_encoder.transform(self.label_encoder.classes_)))
    print(f'[DL] Label map: {label_map}')

    texts = df['cleaned_text'].tolist()

    if cfg.TEST_SIZE > 0:
        # Normal path: train / val / test
        X_temp, X_test, y_temp, y_test = _tts(
            texts, y_cat, test_size=cfg.TEST_SIZE,
            random_state=cfg.RANDOM_SEED, stratify=y)
        val_ratio = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
        X_train, X_val, y_train, y_val = _tts(
            X_temp, y_temp, test_size=val_ratio,
            random_state=cfg.RANDOM_SEED, stratify=y_temp.argmax(axis=1))
        y_test_raw = y_test.argmax(axis=1)
    else:
        # No test split: 99% train / 1% val
        X_train, X_val, y_train, y_val = _tts(
            texts, y_cat, test_size=cfg.VAL_SIZE,
            random_state=cfg.RANDOM_SEED, stratify=y)
        X_test, y_test, y_test_raw = [], np.array([]), np.array([])
        print(f'[DL] No test split — {len(X_train):,} train / {len(X_val):,} val')

    # Tokenize on train only — never fit on val data
    self.tokenizer.fit_on_texts(X_train)
    print(f'[DL] Vocab size: {len(self.tokenizer.word_index):,}')

    def _pad(txts):
        if not txts: return np.array([])
        seqs = self.tokenizer.texts_to_sequences(txts)
        return _pad_seqs(seqs, maxlen=cfg.MAX_SEQ_LEN, padding='post', truncating='post')

    X_train_pad = _pad(X_train)
    X_val_pad   = _pad(X_val)
    X_test_pad  = _pad(X_test) if len(X_test) > 0 else np.array([])

    print(f'[DL] Train: {X_train_pad.shape} | Val: {X_val_pad.shape}')

    return {
        'X_train': X_train_pad, 'y_train': y_train,
        'X_val':   X_val_pad,   'y_val':   y_val,
        'X_test':  X_test_pad,  'y_test':  y_test,
        'y_test_raw': y_test_raw,
        'vocab_size': min(len(self.tokenizer.word_index) + 1, cfg.VOCAB_SIZE + 1),
        'label_encoder': self.label_encoder,
    }

_dlm.DLDataPipeline.prepare = patched_prepare

# ── Run pipeline ──────────────────────────────────────────────────────────
logger.info('=' * 60)
logger.info(f'Training {ARCHITECTURE.upper()} | seq_len={DLConfig.MAX_SEQ_LEN} | vocab={DLConfig.VOCAB_SIZE}')
logger.info('=' * 60)

pipeline = DLDataPipeline(DLConfig)
pipeline.cleaner = patch_emoji_map(pipeline.cleaner)
print('✅ Emoji map patched')

data = pipeline.prepare(DATA_PATH)
pipeline.save(SAVE_DIR)  # saves tokenizer, label_encoder, text_cleaner to Drive

total = len(data['X_train']) + len(data['X_val'])
print(f'\nFinal split:')
print(f'  Train : {len(data["X_train"]):,}  ({len(data["X_train"])/total*100:.1f}%)')
print(f'  Val   : {len(data["X_val"]):,}  ({len(data["X_val"])/total*100:.1f}%)')
print(f'  Test  : none — testing on real YouTube comments in production')
print(f'  Vocab : {data["vocab_size"]:,}')
print(f'  Shape : {data["X_train"].shape}')
print(f'  Saved to Drive ✅')

# ## 10. Train
# 
# **ModelCheckpoint saves to Drive** — if Colab disconnects mid-training, the best checkpoint
# up to that point is already safe in your Drive folder.

# ── Cell 21 ──────────────────────────────────────────────────
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import dl_model as _dlm

start = time.time()

tf.random.set_seed(DLConfig.RANDOM_SEED)
np.random.seed(DLConfig.RANDOM_SEED)

# ── Build model ───────────────────────────────────────────────────────────
vocab_size = data['vocab_size']
model = (build_cnn_bilstm if ARCHITECTURE == 'cnn_bilstm' else build_bilstm_attention)(vocab_size, DLConfig)
model.summary()

# Label smoothing 0.1 — key for generalization on noisy YouTube comments
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer = Adam(learning_rate=DLConfig.LEARNING_RATE, clipnorm=1.0),
    loss      = loss_fn,
    metrics   = ['accuracy'],
)

class_weights = _dlm.compute_class_weights(data['y_train'])
print(f'Class weights: {class_weights}')

# ── Callbacks — checkpoints save directly to Drive ────────────────────────
best_path = os.path.join(SAVE_DIR, 'dl_bilstm_best.keras')
callbacks = [
    WarmupCosineDecay(
        peak_lr       = DLConfig.LEARNING_RATE,
        warmup_epochs = 3,
        total_epochs  = DLConfig.EPOCHS,
        min_lr        = 1e-6,
    ),
    EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 7,
        restore_best_weights = True,
        verbose              = 1,
    ),
    ModelCheckpoint(
        filepath       = best_path,    # saves to Drive
        monitor        = 'val_accuracy',
        save_best_only = True,
        verbose        = 1,
    ),
]

# ── Fit ───────────────────────────────────────────────────────────────────
print(f'\nTraining {ARCHITECTURE} | {len(data["X_train"]):,} comments | seq_len={DLConfig.MAX_SEQ_LEN}...')
print(f'Checkpoints → {best_path}')
history = model.fit(
    data['X_train'], data['y_train'],
    validation_data = (data['X_val'], data['y_val']),
    epochs          = DLConfig.EPOCHS,
    batch_size      = DLConfig.BATCH_SIZE,
    class_weight    = class_weights,
    callbacks       = callbacks,
    verbose         = 1,
)

elapsed = time.time() - start

# ── Save final model to Drive ─────────────────────────────────────────────
final_path = os.path.join(SAVE_DIR, 'dl_bilstm_final.keras')
model.save(final_path)

best_val_acc = max(history.history['val_accuracy'])
best_epoch   = history.history['val_accuracy'].index(best_val_acc) + 1

print(f'\n{"="*60}')
print(f'Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)')
print(f'Best val_accuracy : {best_val_acc:.4f}  (epoch {best_epoch})')
print(f'Total epochs run  : {len(history.history["val_accuracy"])}')
print(f'Model saved → {final_path}')
print(f'{"="*60}')

# ## 11. Training History

# ── Cell 23 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('BiLSTM + Attention — Training History', fontweight='bold', fontsize=14)
epochs = range(1, len(history.history['accuracy']) + 1)

axes[0].plot(epochs, history.history['accuracy'],     'b-o', label='Train', markersize=4)
axes[0].plot(epochs, history.history['val_accuracy'], 'r-o', label='Val',   markersize=4)
axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (ep {best_epoch})')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1)

axes[1].plot(epochs, history.history['loss'],     'b-o', label='Train', markersize=4)
axes[1].plot(epochs, history.history['val_loss'], 'r-o', label='Val',   markersize=4)
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Save to both /content/ (for inline) and Drive (for safety)
plt.savefig('/content/dl_training_history.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{DRIVE_PATH}/dl_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → Drive and /content/')

# ## 12. Validation Set Analysis

# ── Cell 25 ──────────────────────────────────────────────────
print('Running predictions on validation set...')
val_proba = model.predict(data['X_val'], batch_size=DLConfig.BATCH_SIZE, verbose=0)
val_preds = val_proba.argmax(axis=1)
val_true  = data['y_val'].argmax(axis=1)
le        = data['label_encoder']
names     = list(le.classes_)

from sklearn.metrics import classification_report, accuracy_score, f1_score
acc = accuracy_score(val_true, val_preds)
f1  = f1_score(val_true, val_preds, average='weighted')
print(f'\nVal Accuracy : {acc:.4f}')
print(f'Val F1       : {f1:.4f}')
print(f'\n{classification_report(val_true, val_preds, target_names=names, digits=4)}')

cm     = confusion_matrix(val_true, val_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrix (Validation Set)', fontweight='bold')
sns.heatmap(cm,     annot=True, fmt='d',   cmap='Blues',
            xticklabels=names, yticklabels=names, ax=axes[0])
axes[0].set_title('Counts'); axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=names, yticklabels=names, ax=axes[1])
axes[1].set_title('Row % (recall per class)')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('/content/dl_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{DRIVE_PATH}/dl_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Per-class confidence
print('\nPer-class confidence:')
rows = []
for i, name in enumerate(names):
    mc = (val_true == i) & (val_preds == i)
    mw = (val_true == i) & (val_preds != i)
    mt = val_true == i
    rows.append({
        'Class'  : name,
        'Total'  : int(mt.sum()),
        'Recall' : f"{mc.sum()/mt.sum()*100:.1f}%" if mt.sum() > 0 else 'N/A',
        'Conf(correct)': f'{val_proba[mc, i].mean():.3f}' if mc.sum() > 0 else 'N/A',
        'Conf(wrong)':   f'{val_proba[mw, val_preds[mw]].mean():.3f}' if mw.sum() > 0 else 'N/A',
    })
print(pd.DataFrame(rows).to_string(index=False))

errors = [(names[val_true[i]], names[val_preds[i]])
          for i in range(len(val_true)) if val_true[i] != val_preds[i]]
print('\nTop confusion pairs (true → predicted):')
for (t, p), cnt in Counter(errors).most_common(6):
    print(f'  {t:10s} → {p:10s} : {cnt}')

# ## 13. Download
# 
# All model files are already saved to your Drive folder.  
# This cell zips `saved_dl_models/` from Drive and downloads it.
# 
# After downloading:
# 1. Unzip `saved_dl_models.zip`
# 2. Replace **all files** in `backend/saved_dl_models/`
# 3. Restart the backend
# 
# > ⚠️ Always replace `dl_tokenizer.pkl` and `dl_text_cleaner.pkl` together with the model.
# > Using an old tokenizer with a new model gives garbage predictions.

# ── Cell 27 ──────────────────────────────────────────────────

import shutil, zipfile

# Zip the saved_dl_models folder from Drive
shutil.make_archive('/content/saved_dl_models', 'zip', SAVE_DIR)

print('Files in zip:')
with zipfile.ZipFile('/content/saved_dl_models.zip', 'r') as z:
    for name in z.namelist():
        info = z.getinfo(name)
        print(f'  {name:<45} {info.file_size/1024/1024:.2f} MB')

files.download('/content/saved_dl_models.zip')
print('\n✅ Done. Replace backend/saved_dl_models/ and restart the backend.')
print(f'\nModels also saved permanently to: {SAVE_DIR}')
