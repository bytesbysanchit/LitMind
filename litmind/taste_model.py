"""Reading-taste model: "will I like this book?" with (ridge) linear regression.

Data format
-----------
Both the catalogue and your reading history use the SAME columns as Books.csv:

    ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L

Your history CSV may optionally have one extra column, `My-Rating` (any 1-5 / 1-10 / 1-100 scale).

How the regression is set up
----------------------------
Books.csv has no ratings, so the target is built from what we know:

    books you read             -> 1.0   (or rating / max rating if My-Rating is given)
    random catalogue books     -> 0.0   ("not in my history"; sampled, NOT books you disliked)

A linear model on TF-IDF title words, author and publisher then predicts a score for any
book. Because the negatives are only "unread", the score means "looks like what I read", and it is
reported as a percentile against the catalogue rather than as a probability.

We use Ridge (linear regression + L2 penalty): with thousands of sparse text features and a few
dozen books, plain least squares would memorise the training set.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

COLUMNS = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher",
           "Image-URL-S", "Image-URL-M", "Image-URL-L"]
REQUIRED = ["Book-Title", "Book-Author"]
ALPHAS = [0.3, 1, 3, 10, 30, 100]


class FormatError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_books(source, *, validate_format: bool = True) -> pd.DataFrame:
    """Read a Books.csv-format file (path or file-like) and clean it."""
    df = pd.read_csv(source, low_memory=False, encoding="latin-1", on_bad_lines="skip", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise FormatError(
            f"Missing column(s) {missing}. Expected the Books.csv format: {', '.join(COLUMNS)}")
    if validate_format:
        absent = [c for c in COLUMNS if c not in df.columns]
        for c in absent:                       # tolerate missing optional columns
            df[c] = ""
    return clean_books(df)


def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Book-Title", "Book-Author", "Publisher"):
        df[c] = df[c].fillna("").astype(str).map(lambda s: html.unescape(s).strip())
    year = pd.to_numeric(df.get("Year-Of-Publication"), errors="coerce")
    df["Year"] = year.where((year >= 1400) & (year <= 2026))   # 0 and 2030+ are junk in Books.csv
    df["_author"] = df["Book-Author"].str.casefold().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    df["_publisher"] = df["Publisher"].str.casefold().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df["_title"] = df["Book-Title"].str.replace(r"\(.*?\)", " ", regex=True)   # drop "(Large Print)" etc.
    df["_decade"] = (df["Year"] // 10 * 10).map(lambda y: "unk" if pd.isna(y) else f"d{int(y)}")
    df["_key"] = (df["Book-Title"].str.casefold().str.replace(r"[^a-z0-9]", "", regex=True) + "|" + df["_author"])
    return df.reset_index(drop=True)


def make_row(title: str, author: str = "", year: int | None = None, publisher: str = "") -> pd.DataFrame:
    return clean_books(pd.DataFrame([{
        "ISBN": "", "Book-Title": title, "Book-Author": author,
        "Year-Of-Publication": "" if year is None else str(year), "Publisher": publisher,
        "Image-URL-S": "", "Image-URL-M": "", "Image-URL-L": ""}]))


def _features() -> ColumnTransformer:
    return ColumnTransformer([
        ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                  max_features=30000, token_pattern=r"(?u)\b[^\W\d_]{2,}\b"), "_title"),
        ("author", OneHotEncoder(handle_unknown="ignore"), ["_author"]),
        ("publisher", OneHotEncoder(handle_unknown="ignore"), ["_publisher"]),
    ], sparse_threshold=1.0)          # always sparse (dense output would break explain())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class TasteModel:
    pipeline: ColumnTransformer | None = None
    ridge: Ridge | None = None
    alpha: float = 1.0
    reference_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    report: dict = field(default_factory=dict)

    # ----------------------------------------------------------- training
    def fit(self, history: pd.DataFrame, catalog: pd.DataFrame, *, neg_ratio: int = 5,
            seed: int = 42) -> "TasteModel":
        if len(history) < 5:
            raise ValueError("Add at least 5 books to your reading history (15+ works much better).")
        history = history.drop_duplicates("_key").reset_index(drop=True)
        y_pos = _targets(history)
        unread = catalog[~catalog["_key"].isin(set(history["_key"]))]
        if "ISBN" in history and "ISBN" in unread:
            unread = unread[~unread["ISBN"].isin(set(history["ISBN"]) - {""})]
        n_neg = min(len(unread), max(50, neg_ratio * len(history)))
        neg = unread.sample(n_neg, random_state=seed)

        data = pd.concat([history, neg], ignore_index=True)
        y = np.concatenate([y_pos, np.zeros(len(neg))])
        label = np.concatenate([np.ones(len(history)), np.zeros(len(neg))]).astype(int)

        self.report = self._cross_validate(data, y, label, seed)
        self.alpha = self.report.get("best_alpha", 10.0)
        self.pipeline = _features()
        X = self.pipeline.fit_transform(data)
        self.ridge = Ridge(alpha=self.alpha).fit(X, y)

        held_out = unread.drop(neg.index)          # never seen in training: unbiased reference
        ref = held_out.sample(min(len(held_out), 5000), random_state=seed + 1)
        self.reference_scores = np.sort(self._raw(ref))
        self.report.update(n_history=int(len(history)), n_negatives=int(len(neg)),
                           n_features=int(X.shape[1]))
        return self

    def _cross_validate(self, data, y, label, seed) -> dict:
        n_pos = int(label.sum())
        folds = min(5, n_pos)
        if folds < 3:
            return {"cv": "skipped (too few books)", "best_alpha": 10.0}
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        best = None
        for alpha in ALPHAS:
            pred = np.zeros(len(y))
            for tr, te in skf.split(data, label):
                feats = _features()
                Xtr = feats.fit_transform(data.iloc[tr])
                pred[te] = Ridge(alpha=alpha).fit(Xtr, y[tr]).predict(feats.transform(data.iloc[te]))
            auc = roc_auc_score(label, pred)
            if best is None or auc > best["auc"]:
                best = {"alpha": alpha, "auc": auc, "pred": pred.copy()}
        return {
            "cv_folds": folds,
            "best_alpha": best["alpha"],
            "cv_auc": round(float(best["auc"]), 3),
            "cv_r2": round(float(r2_score(y, best["pred"])), 3),
            "cv_mae": round(float(mean_absolute_error(y, np.clip(best["pred"], 0, 1))), 3),
        }

    # ------------------------------------------------------------ scoring
    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.ridge.predict(self.pipeline.transform(df))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Percentile (0-100) of each book's score against typical catalogue books."""
        raw = self._raw(df)
        return 100.0 * np.searchsorted(self.reference_scores, raw, side="right") / len(self.reference_scores)

    def predict_one(self, title: str, author: str = "", year: int | None = None,
                    publisher: str = "", top_k: int = 6) -> dict:
        row = make_row(title, author, year, publisher)
        raw = float(self._raw(row)[0])
        pct = float(self.score(row)[0])
        known = int(self.pipeline.transform(row).nnz)
        # Features the model has never seen contribute nothing: the score is then just the
        # intercept, which says "no evidence", not "good match".
        informative = known >= 1 and abs(raw - float(self.ridge.intercept_)) > 1e-6
        return {
            "raw_score": round(raw, 4),
            "percentile": round(pct, 1),
            "known_features": known,
            "verdict": verdict(pct) if informative else "Not enough information",
            "reasons": self.explain(row, top_k),
        }

    def explain(self, row: pd.DataFrame, top_k: int = 6) -> dict:
        x = self.pipeline.transform(row)
        names = self.pipeline.get_feature_names_out()
        contrib = x.multiply(self.ridge.coef_).toarray().ravel()
        order = np.argsort(contrib)
        pretty = lambda n: re.sub(r"^(title|author|publisher)__", lambda m: m.group(1) + ": ", n)
        up = [(pretty(names[i]), round(float(contrib[i]), 4)) for i in order[::-1][:top_k] if contrib[i] > 1e-4]
        down = [(pretty(names[i]), round(float(contrib[i]), 4)) for i in order[:top_k] if contrib[i] < -1e-4]
        return {"raises_match": up, "lowers_match": down}

    def recommend(self, catalog: pd.DataFrame, history: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        pool = catalog[~catalog["_key"].isin(set(history["_key"]))].copy()
        pool["Match-Percentile"] = self.score(pool)
        pool = pool.sort_values("Match-Percentile", ascending=False).drop_duplicates("_key")
        return pool.head(top_n)[["Book-Title", "Book-Author", "Year", "Publisher", "ISBN", "Match-Percentile"]]

    # -------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TasteModel":
        return joblib.load(path)


def _targets(history: pd.DataFrame) -> np.ndarray:
    if "My-Rating" not in history.columns:
        return np.ones(len(history))
    r = pd.to_numeric(history["My-Rating"], errors="coerce")
    if r.notna().sum() == 0:
        return np.ones(len(history))
    top = r.max()
    scale = 5 if top <= 5 else 10 if top <= 10 else 100
    return (r.fillna(top) / scale).clip(0, 1).to_numpy()


def verdict(percentile: float) -> str:
    if percentile >= 90:
        return "Strong match"
    if percentile >= 70:
        return "Good match"
    if percentile >= 40:
        return "Possible match"
    return "Weak match"
