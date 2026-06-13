"""
SecureLink XGBoost Demo (DUMMY DATA)
====================================
A safe, self-contained demo of the privacy-preserving record linkage pipeline
I built for the COMP3850 SecureLink project at Macquarie University.

This script uses RANDOMLY GENERATED FAKE DATA only. It contains no real
datasets, no confidential code, and no personal information. Its purpose is to
demonstrate the *approach* and to print the kind of technical details an
interviewer asks about (number of data points, number of classes, class
balance, train/test split).

It trains the same three model configurations as the real project:
  1. Raw XGBoost        - no privacy, baseline
  2. LDP XGBoost        - Local Differential Privacy (noise added to features)
  3. Bloom Filter XGB   - records hashed to bits, compared by Dice similarity

The real project's headline result (on the real dataset) was:
  adding LDP privacy cost only ~1.7% F1 vs the raw baseline.
The numbers below come from FAKE data, so they are illustrative only.

Run:  python securelink_xgboost_demo.py
Needs: pip install xgboost scikit-learn pandas numpy matplotlib
"""

import hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # save charts to file without a screen
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Privacy budget. Smaller epsilon = more noise = more privacy.
EPSILON = 7.0


# ---------------------------------------------------------------------------
# 1. Make fake "record pair" data
# ---------------------------------------------------------------------------
# Each row is a PAIR of short clinical-style notes (text1, text2) plus a label:
#   label = 1  -> the two notes describe the SAME person (a true match)
#   label = 0  -> the two notes describe DIFFERENT people (a non-match)
# Real linkage data is heavily imbalanced (far more non-matches), so we
# generate many more 0s than 1s on purpose, to show how that is handled.

CONDITIONS = ["hypertension", "type 2 diabetes", "asthma", "migraine",
              "anaemia", "arthritis", "eczema", "anxiety", "reflux",
              "fracture", "bronchitis", "sinusitis"]
MEDS = ["metformin", "ibuprofen", "ventolin", "atorvastatin", "amoxicillin",
        "paracetamol", "sertraline", "omeprazole", "salbutamol", "aspirin"]
NOTE_WORDS = ["patient", "presents", "with", "history", "of", "prescribed",
              "follow", "up", "review", "stable", "mild", "chronic", "acute"]


def make_note(condition, med):
    """Build a short fake note from a condition + medication + filler words."""
    filler = np.random.choice(NOTE_WORDS, size=np.random.randint(4, 8))
    words = ["patient", "with", condition, "prescribed", med] + list(filler)
    np.random.shuffle(words)
    return " ".join(words)


def add_typos(text):
    """Corrupt a note so 'same person' pairs are clearly NOT identical.
    Drops and swaps several words, the way messy real notes differ."""
    words = text.split()
    # drop ~20% of the words
    keep = [w for w in words if np.random.rand() > 0.2]
    if len(keep) < 2:
        keep = words[:2]
    # shuffle the survivors
    np.random.shuffle(keep)
    return " ".join(keep)


def generate_pairs(n_matches, n_non_matches):
    """Create a dataframe of fake record pairs with text1, text2, label, uid."""
    rows = []
    uid = 0
    # True matches: two MESSY versions of the same underlying note
    for _ in range(n_matches):
        cond, med = np.random.choice(CONDITIONS), np.random.choice(MEDS)
        base = make_note(cond, med)
        rows.append({"uid": uid, "text1": base,
                     "text2": add_typos(base), "label": 1})
        uid += 1
    # Non-matches: half are easy (unrelated), half are HARD (share a term),
    # which makes the similarity score overlap with real matches.
    for _ in range(n_non_matches):
        if np.random.rand() < 0.3:
            shared = np.random.choice(CONDITIONS)   # share a condition
            c1 = c2 = shared
            m1, m2 = np.random.choice(MEDS), np.random.choice(MEDS)
        else:
            c1, c2 = np.random.choice(CONDITIONS), np.random.choice(CONDITIONS)
            m1, m2 = np.random.choice(MEDS), np.random.choice(MEDS)
        rows.append({"uid": uid, "text1": make_note(c1, m1),
                     "text2": make_note(c2, m2), "label": 0})
        uid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED)
    return df.reset_index(drop=True)


# Imbalanced on purpose: ~5x more non-matches than matches
df_train = generate_pairs(n_matches=300, n_non_matches=1000)
df_test = generate_pairs(n_matches=120, n_non_matches=400)


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def evaluate(model, X, y_true, name):
    """Print the four standard metrics and return them."""
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {pre:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    return acc, pre, rec, f1


def apply_ldp_noise(X, epsilon):
    """Local Differential Privacy: add small random noise to the features.
    Smaller epsilon -> higher flip probability -> more privacy."""
    flip_prob = 1.0 / (1.0 + np.exp(epsilon / 2.0))
    noise = np.random.binomial(1, flip_prob, size=X.shape)
    sign = np.random.binomial(1, 0.5, size=X.shape)
    return X + noise * ((-1) ** sign) * 0.1


def text_to_bloom(text, length=512, num_hash=10, q=2):
    """Encode a string into a Bloom filter (a privacy-preserving bit vector).
    The text is split into q-grams, each hashed into bit positions."""
    bits = np.zeros(length, dtype=int)
    grams = [text[i:i + q] for i in range(len(text) - q + 1)]
    for g in grams:
        h1 = int(hashlib.sha1(g.encode()).hexdigest(), 16)
        h2 = int(hashlib.md5(g.encode()).hexdigest(), 16)
        for i in range(num_hash):
            bits[(h1 + i * h2) % length] = 1
    return bits


def add_bit_noise(bits, flip_prob):
    """Flip a fraction of bits for differential privacy on the Bloom filter."""
    mask = np.random.binomial(1, flip_prob, size=bits.shape)
    return bits ^ mask


def dice_similarity(b1, b2):
    """Dice coefficient: how much two bit vectors overlap (0 to 1)."""
    common = np.sum(b1 & b2)
    total = np.sum(b1) + np.sum(b2)
    return (2.0 * common / total) if total > 0 else 0.0


def render_sample_table(rows, filename):
    """Render sample record pairs (Type, Note A, Note B, Similarity) as a PNG."""
    def short(t, n=38):
        return t if len(t) <= n else t[:n - 1] + "…"

    col_labels = ["Type", "Note A (text1)", "Note B (text2)", "Dice sim"]
    cell_text = [[r["type"], short(r["a"]), short(r["b"]), f"{r['sim']:.3f}"]
                 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.45 * len(rows)))
    ax.axis("off")
    ax.set_title("SecureLink: sample matches and non-matches (DUMMY DATA)",
                 pad=12)
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    # colour the header and tint match vs non-match rows
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f3b52")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            is_match = cell_text[r - 1][0] == "Match"
            cell.set_facecolor("#e8f4ea" if is_match else "#f4eae8")
    tbl.auto_set_column_width([0, 1, 2, 3])
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Feature pipeline: text -> numbers (TF-IDF then Truncated SVD)
# ---------------------------------------------------------------------------
print("=" * 60)
print("DATA SUMMARY (fake data)")
print("=" * 60)
print(f"Training pairs : {len(df_train)}")
print(f"Test pairs     : {len(df_test)}")
print(f"Target classes : 2  (1 = match, 0 = non-match)")
print(f"Train class counts (non-match, match): "
      f"{np.bincount(df_train['label'])}")
print(f"Test  class counts (non-match, match): "
      f"{np.bincount(df_test['label'])}")
print()

vectorizer = TfidfVectorizer(stop_words="english")
Xtr1 = vectorizer.fit_transform(df_train["text1"])
Xtr2 = vectorizer.transform(df_train["text2"])
Xte1 = vectorizer.transform(df_test["text1"])
Xte2 = vectorizer.transform(df_test["text2"])

# Real project used 100 components per field. Fake data has a small
# vocabulary, so cap the components to what the data can support.
n_comp = min(50, Xtr1.shape[1] - 1, Xtr2.shape[1] - 1)
svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
X_train = np.hstack([svd.fit_transform(Xtr1), svd.transform(Xtr2)])
X_test = np.hstack([svd.transform(Xte1), svd.transform(Xte2)])
y_train = df_train["label"].astype(int).values
y_test = df_test["label"].astype(int).values
print(f"Feature count for Raw/LDP models: {X_train.shape[1]} "
      f"(SVD components from both text fields; real project used 200)")
print()


# ---------------------------------------------------------------------------
# 4. Model A: Raw XGBoost (no privacy) - the baseline
# ---------------------------------------------------------------------------
print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)
model_raw = xgb.XGBClassifier(eval_metric="logloss")
model_raw.fit(X_train, y_train)
acc1, p1, r1, f1_1 = evaluate(model_raw, X_test, y_test, "Raw XGBoost (no privacy)")
print()


# ---------------------------------------------------------------------------
# 5. Model B: LDP XGBoost (privacy by adding noise)
# ---------------------------------------------------------------------------
X_train_ldp = apply_ldp_noise(X_train, EPSILON)
X_test_ldp = apply_ldp_noise(X_test, EPSILON)
model_ldp = xgb.XGBClassifier(eval_metric="logloss")
model_ldp.fit(X_train_ldp, y_train)
acc2, p2, r2, f1_2 = evaluate(model_ldp, X_test_ldp, y_test,
                              f"LDP XGBoost (epsilon={EPSILON})")
print()


# ---------------------------------------------------------------------------
# 6. Model C: Bloom Filter XGBoost (privacy by hashing)
# ---------------------------------------------------------------------------
# Encode every record into a Bloom filter, add DP bit-flip noise, then use the
# Dice similarity between the two notes in each pair as a SINGLE feature.
# The real project added DP noise plus extra bit flips; we use a visible flip
# rate here so the privacy cost shows up the way it did in the real results.
bf_flip_prob = 0.06
df_all = pd.concat([df_train, df_test], ignore_index=True)

sims, labels = [], []
pair_rows = []
for _, row in df_all.iterrows():
    b1 = add_bit_noise(text_to_bloom(row["text1"]), bf_flip_prob)
    b2 = add_bit_noise(text_to_bloom(row["text2"]), bf_flip_prob)
    sim = dice_similarity(b1, b2)
    sims.append([sim])
    labels.append(row["label"])
    pair_rows.append({"type": "Match" if row["label"] == 1 else "Non-match",
                      "a": row["text1"], "b": row["text2"], "sim": sim})

X_bf = np.array(sims)
y_bf = np.array(labels)
print(f"  Bloom Filter dataset before balancing: "
      f"{np.bincount(y_bf)}  (non-match, match)")

# Handle class imbalance: keep all matches, keep 3x as many non-matches (3:1)
pos_idx = np.where(y_bf == 1)[0]
neg_idx = np.where(y_bf == 0)[0]
neg_keep = np.random.choice(neg_idx, size=min(len(pos_idx) * 3, len(neg_idx)),
                            replace=False)
keep = np.concatenate([pos_idx, neg_keep])
X_bf, y_bf = X_bf[keep], y_bf[keep]
print(f"  Bloom Filter dataset after balancing : "
      f"{np.bincount(y_bf)}  (non-match, match)")

Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    X_bf, y_bf, test_size=0.3, random_state=RANDOM_SEED)
model_bf = xgb.XGBClassifier(eval_metric="logloss")
model_bf.fit(Xb_tr, yb_tr)
print()
acc3, p3, r3, f1_3 = evaluate(model_bf, Xb_te, yb_te,
                              "Bloom Filter XGBoost (Dice similarity feature)")
print()


# ---------------------------------------------------------------------------
# 7. Comparison chart
# ---------------------------------------------------------------------------
labels = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(labels))
w = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w, [acc1, p1, r1, f1_1], w, label="Raw")
ax.bar(x,     [acc2, p2, r2, f1_2], w, label="LDP")
ax.bar(x + w, [acc3, p3, r3, f1_3], w, label="Bloom Filter")
ax.set_ylim(0, 1.15)            # headroom so labels clear the top border
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("Score")
ax.set_title("SecureLink XGBoost: privacy vs accuracy (DUMMY DATA)",
             pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="lower left")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("securelink_demo_chart.png", dpi=120)
print("Saved chart to securelink_demo_chart.png")

# Sample 5 matches and 5 non-matches (like the app's MatchViewer)
sample_matches = [r for r in pair_rows if r["type"] == "Match"][:5]
sample_non_matches = [r for r in pair_rows if r["type"] == "Non-match"][:5]
render_sample_table(sample_matches + sample_non_matches,
                    "securelink_sample_matches.png")
print("Saved sample table to securelink_sample_matches.png")
print()
print("NOTE: numbers above are from FAKE data and are illustrative only.")
print("The real project result was ~1.7% F1 drop from Raw to LDP at epsilon=7.")