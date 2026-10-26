"""Experiments for the taste model (all numbers in the report come from this script).

  python scripts/evaluate_taste.py data/Books.csv data/samples/my_read_books.csv
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from litmind.taste_model import TasteModel, load_books

def feats(which):
    parts = {"title": ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                       max_features=30000, token_pattern=r"(?u)\b[^\W\d_]{2,}\b"), "_title"),
             "author": ("author", OneHotEncoder(handle_unknown="ignore"), ["_author"]),
             "publisher": ("publisher", OneHotEncoder(handle_unknown="ignore"), ["_publisher"]),
             "year": ("year", OneHotEncoder(handle_unknown="ignore"), ["_decade"])}
    return ColumnTransformer([parts[w] for w in which], sparse_threshold=1.0)

def make_data(hist, cat, neg_ratio, seed):
    unread = cat[~cat["_key"].isin(set(hist["_key"]))]
    neg = unread.sample(min(len(unread), neg_ratio * len(hist)), random_state=seed)
    data = pd.concat([hist, neg], ignore_index=True)
    y = np.r_[np.ones(len(hist)), np.zeros(len(neg))]
    return data, y

def cv_auc(hist, cat, which=("title", "author", "publisher"), alpha=3.0, neg_ratio=5, seed=0, model="ridge", rule=None):
    data, y = make_data(hist, cat, neg_ratio, seed)
    k = min(5, int(y.sum())); pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(data, y):
        if rule == "author_seen":                       # trivial heuristic: has this author been read?
            seen = set(data.iloc[tr][y[tr] == 1]["_author"])
            pred[te] = data.iloc[te]["_author"].isin(seen).astype(float) + 1e-6 * np.random.default_rng(seed).random(len(te))
            continue
        f = feats(which); Xtr = f.fit_transform(data.iloc[tr])
        m = Ridge(alpha=alpha) if model == "ridge" else LinearRegression()
        pred[te] = m.fit(Xtr, y[tr]).predict(f.transform(data.iloc[te]))
    return roc_auc_score(y, pred)

def multi(fn, seeds=range(5)):
    v = [fn(s) for s in seeds]; return {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}

if __name__ == "__main__":
    cat = load_books(sys.argv[1]); hist = load_books(sys.argv[2]); R = {}
    print("catalogue", len(cat), "history", len(hist))
    R["feature_sets"] = {"+".join(w): multi(lambda s, w=w: cv_auc(hist, cat, w, seed=s))
                         for w in [("title",), ("author",), ("publisher",), ("title", "author"), ("title", "author", "publisher"),
                                   ("title", "author", "publisher", "year")]}
    R["baselines"] = {"random": {"mean": 0.5, "std": 0.0},
                      "author_seen_rule": multi(lambda s: cv_auc(hist, cat, seed=s, rule="author_seen"))}
    R["alpha_sweep"] = {str(a): multi(lambda s, a=a: cv_auc(hist, cat, alpha=a, seed=s)) for a in [0.1, 0.3, 1, 3, 10, 30, 100, 300]}
    R["ols_vs_ridge"] = {"ols": multi(lambda s: cv_auc(hist, cat, model="ols", seed=s)), "ridge_alpha3": multi(lambda s: cv_auc(hist, cat, seed=s))}
    R["neg_ratio"] = {str(r): multi(lambda s, r=r: cv_auc(hist, cat, neg_ratio=r, seed=s)) for r in [1, 2, 5, 10, 20]}
    rng = np.random.default_rng(0); R["history_size"] = {}
    for n in [8, 12, 18, 24, 30, len(hist)]:
        R["history_size"][str(n)] = multi(lambda s, n=n: cv_auc(hist.sample(n, random_state=s) if n < len(hist) else hist, cat, seed=s))
    # leave-one-book-out and leave-one-author-out: where does the hidden book rank among catalogue books?
    def hidden_rank(group_by_author):
        pcts = []
        for i in range(len(hist)):
            drop = hist["_author"] == hist.iloc[i]["_author"] if group_by_author else hist.index == i
            m = TasteModel().fit(hist[~drop].reset_index(drop=True), cat, alpha=3.0)
            held = hist[hist.index == i] if not group_by_author else hist[hist.index == i]
            pcts.append(float(m.score(held)[0]))
        a = np.array(pcts); return {"median_percentile": round(float(np.median(a)), 1), "share_top10pct": round(float((a >= 90).mean()), 3),
                                    "share_top50pct": round(float((a >= 50).mean()), 3), "n": len(a), "percentiles": [round(x, 1) for x in pcts]}
    R["leave_one_book_out"] = hidden_rank(False); R["leave_one_author_out"] = hidden_rank(True)
    # runtime
    t = time.time(); m = TasteModel().fit(hist, cat); R["fit_seconds_with_cv"] = round(time.time() - t, 1); R["report"] = m.report
    t = time.time(); m.recommend(cat, hist, 20); R["recommend_seconds"] = round(time.time() - t, 1)
    json.dump(R, open(Path(__file__).resolve().parents[1] / "reports" / "taste_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("leave_one_book_out", "leave_one_author_out")}, indent=1))
    for k in ("leave_one_book_out", "leave_one_author_out"):
        print(k, {a: b for a, b in R[k].items() if a != "percentiles"})
