"""Hit Predictor: the chance that a game becomes a hit, from choices made before launch.

Run from the project root (retrains and overwrites models/):
    python -m src.model

Writes:
    models/hit_model.joblib   calibrated model + the plain logistic regression used to explain it
    models/metrics.json       every number reported in the notebook and dashboard
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, learning_curve,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import analysis as A
from src import config as C
from src.cleaning import load_clean

SEED = 42
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]
OVERFIT_GAP = 0.03            # flag if train ROC-AUC is this much above test ROC-AUC

# Pre-launch choices only. Deliberately NOT used:
#  - n_languages, achievements, has_dlc: often added or expanded after launch, when a game is
#    already successful (reverse causality).
#  - playtime, review_count, rating_pct, owners, peak_ccu: measured after launch.
#  - visibility_pctile, quadrant, is_hit_*: built from review_count, i.e. from the target itself
#    (leakage).
# Price enters as bands (Free, Under $5, $5–10, $10–20, $20+), one-hot, not as a straight line:
# a linear price term made the chance rise with every extra dollar. Price mostly stands for budget
# and scope (a confounder), so it is an association, not a lever. "Free" replaces is_free.
NUMERIC = ["n_platforms"]
BINARY = ["is_indie", "is_early_access", "is_multiplayer"]
FEATURES = NUMERIC + BINARY + ["price_band", "release_month", "genres_core", "subgenres"]
TARGET = "is_hit"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- pipeline pieces
def split_labels(text):
    """'Action|RPG' -> ['Action', 'RPG'] (multi-label text columns)."""
    return [t for t in str(text).split("|") if t]


def multi_hot():
    """One column per genre / sub-genre, vocabulary learned from the training data."""
    return CountVectorizer(tokenizer=split_labels, token_pattern=None, lowercase=False,
                           binary=True)


def make_preprocessor():
    return ColumnTransformer([
        ("platforms", StandardScaler(), ["n_platforms"]),
        ("binary", "passthrough", BINARY),
        ("price", OneHotEncoder(categories=[C.PRICE_LABELS], handle_unknown="ignore"),
         ["price_band"]),
        ("month", OneHotEncoder(handle_unknown="ignore"), ["release_month"]),
        ("genre", multi_hot(), "genres_core"),
        ("subgenre", multi_hot(), "subgenres"),
    ])


def make_logreg():
    # class_weight="balanced" handles the 5% hit rate; probabilities are calibrated afterwards
    return Pipeline([("prep", make_preprocessor()),
                     ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))])


def cv_splitter():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ---------------------------------------------------------------- data
def load_data(df=None):
    """Study cohort 2020–2023 with the model features."""
    df = load_clean() if df is None else df
    s = A.study(df).copy()
    for col in BINARY:
        s[col] = s[col].astype(int)
    s["price_band"] = s["price_band"].astype(str)
    s["genres_core"] = s["genres_core"].fillna("")
    s["subgenres"] = s["subgenres"].fillna("")
    return s


def split(s):
    """Stratified 80/20 split. The test part is only used once, at the very end."""
    return train_test_split(s, test_size=0.2, stratify=s[TARGET], random_state=SEED)


def tune(train, model=None, grid=None):
    """5-fold stratified CV on the training data only (ROC-AUC)."""
    search = GridSearchCV(model or make_logreg(), grid or {"clf__C": C_GRID},
                          scoring="roc_auc", cv=cv_splitter(), return_train_score=True,
                          n_jobs=-1)
    search.fit(train[FEATURES], train[TARGET])
    return search


def calibrate(best_pipeline, train):
    """Turn balanced-class scores into realistic probabilities (fitted inside the training data)."""
    cal = CalibratedClassifierCV(best_pipeline, method="sigmoid", cv=cv_splitter())
    cal.fit(train[FEATURES], train[TARGET])
    return cal


# ---------------------------------------------------------------- explanation
def feature_names(pipeline):
    prep = pipeline.named_steps["prep"]
    names = ["Platforms", "Indie label", "Early Access", "Multiplayer"]
    names += [f"Price band: {b}" for b in prep.named_transformers_["price"].categories_[0]]
    names += [f"Release month: {MONTHS[int(m) - 1]}"
              for m in prep.named_transformers_["month"].categories_[0]]
    names += [f"Steam genre: {g}" for g in prep.named_transformers_["genre"].get_feature_names_out()]
    names += [f"Sub-genre: {g}"
              for g in prep.named_transformers_["subgenre"].get_feature_names_out()]
    return names


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def coefficients(pipeline):
    """Logistic-regression coefficients per (transformed) feature, largest effect first."""
    coef = pipeline.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"feature": feature_names(pipeline), "coef": coef})
    return t.reindex(t["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def explain(bundle, row):
    """What moved the score for one input row: coef × (value − training mean), on the log-odds
    scale, only for the inputs the user actually chose (selected genres, month, etc.)."""
    pipe = bundle["explainer"]
    x = _dense(pipe.named_steps["prep"].transform(row[FEATURES]))[0]
    contrib = pipe.named_steps["clf"].coef_[0] * (x - bundle["feature_means"])
    names = bundle["feature_names"]
    keep = [i for i, n in enumerate(names)
            if not n.startswith(("Price band", "Release month", "Steam genre", "Sub-genre"))
            or x[i] == 1]
    keep = [i for i in keep if names[i] != "Indie label"]      # fixed for every Indie input
    r = row.iloc[0]
    state = {"Platforms": f"Platforms: {int(r['n_platforms'])}",
             "Early Access": f"Early Access: {'yes' if r['is_early_access'] else 'no'}",
             "Multiplayer": f"Multiplayer: {'yes' if r['is_multiplayer'] else 'no'}"}
    t = pd.DataFrame({"factor": [state.get(names[i], names[i]) for i in keep],
                      "effect": contrib[keep]})
    return t.sort_values("effect", ascending=False)


def predict(bundle, row):
    return float(bundle["model"].predict_proba(row[FEATURES])[0, 1])


def make_row(price_band, n_platforms, multiplayer, early_access, month, genres=(), subgenres=(),
             indie=True):
    """One dashboard input -> a one-row DataFrame in the model's format."""
    return pd.DataFrame([{
        "price_band": str(price_band), "n_platforms": int(n_platforms), "is_indie": int(indie),
        "is_early_access": int(early_access), "is_multiplayer": int(multiplayer),
        "release_month": int(month), "genres_core": "|".join(genres),
        "subgenres": "|".join(subgenres)}])


# ---------------------------------------------------------------- game layer (dashboard)
# Ranks compare an idea with every Indie game in the 2020–2023 cohort (percentile of the
# predicted chance), so a small % still reads clearly: S = top 5%, A = top 20%, ...
RANKS = [("S", 95), ("A", 80), ("B", 60), ("C", 30), ("D", 0)]


def indie_scores(bundle, df=None):
    """Sorted predicted chances of all Indie games in the study cohort (the ranking scale)."""
    s = load_data(df)
    s = s[s["is_indie"] == 1]
    return np.sort(bundle["model"].predict_proba(s[FEATURES])[:, 1])


def percentile(p, scores):
    """Share of Indie games (in %) with a lower predicted chance than p."""
    return 100 * np.searchsorted(scores, p, side="left") / len(scores)


def rank_of(pct):
    return next(r for r, cut in RANKS if pct >= cut)


def predict_many(bundle, ideas):
    rows = pd.concat([make_row(**i) for i in ideas], ignore_index=True)
    return bundle["model"].predict_proba(rows[FEATURES])[:, 1]


def power_ups(bundle, idea, sub_options, min_gain=0.5, n=3):
    """One-step changes to an idea that raise the predicted chance the most (in % points).

    Price bands are left out on purpose: price stands for budget, so "raise the price" would
    be a false lever (see the caveat in the dashboard).
    """
    base = idea
    variants = []
    subs = list(base["subgenres"])
    for i, old in enumerate(subs):
        for new in sub_options:
            if new not in subs:
                variants.append((f"swap {old} for {new}",
                                 {**base, "subgenres": subs[:i] + [new] + subs[i + 1:]}))
    if len(subs) < 3:
        for new in sub_options:
            if new not in subs:
                variants.append((f"add {new}", {**base, "subgenres": subs + [new]}))
    variants.append(("drop Multiplayer" if base["multiplayer"] else "add Multiplayer",
                     {**base, "multiplayer": not base["multiplayer"]}))
    variants.append(("skip Early Access" if base["early_access"] else "launch in Early Access",
                     {**base, "early_access": not base["early_access"]}))
    for m in range(1, 13):
        if m != base["month"]:
            variants.append((f"release in {MONTHS[m - 1]}", {**base, "month": m}))
    p0 = predict_many(bundle, [base])[0]
    p = predict_many(bundle, [v for _, v in variants])
    gains = pd.DataFrame({"change": [c for c, _ in variants], "p": p, "gain_pts": (p - p0) * 100})
    return gains[gains["gain_pts"] >= min_gain].sort_values("gain_pts", ascending=False).head(n)


def random_idea(df, rng, sub_options, genre_options):
    """A random but realistic idea: the profile of a real Indie game from the cohort."""
    s = load_data(df)
    s = s[(s["is_indie"] == 1) & (s["subgenres"] != "")]
    while True:
        g = s.iloc[int(rng.integers(len(s)))]
        subs = [x for x in split_labels(g["subgenres"]) if x in sub_options]
        gens = [x for x in split_labels(g["genres_core"]) if x in genre_options]
        if subs and gens:
            break
    k = int(rng.integers(1, min(3, len(subs)) + 1))
    return {"price_band": g["price_band"], "n_platforms": int(g["n_platforms"]),
            "multiplayer": bool(g["is_multiplayer"]), "early_access": bool(g["is_early_access"]),
            "month": int(g["release_month"]),
            "genres": [gens[int(rng.integers(len(gens)))]],
            "subgenres": [str(x) for x in rng.choice(subs, size=k, replace=False)]}


def load_model():
    return joblib.load(C.MODEL_FILE), json.loads(C.METRICS_FILE.read_text())


# ---------------------------------------------------------------- evaluation helpers
def test_metrics(y, p, threshold):
    top = p >= np.quantile(p, 0.9)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold).ravel()
    return {"roc_auc": round(roc_auc_score(y, p), 3),
            "average_precision": round(average_precision_score(y, p), 3),
            "brier": round(brier_score_loss(y, p), 4),
            "base_rate_pct": round(100 * y.mean(), 1),
            "top10_hit_rate_pct": round(100 * y[top].mean(), 1),
            "threshold": round(float(threshold), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "precision_at_threshold": round(tp / max(tp + fp, 1), 3),
            "recall_at_threshold": round(tp / max(tp + fn, 1), 3)}


def time_check(s):
    """Generalisation over time: tune + calibrate on 2020–2022, test on 2023."""
    past, future = s[s["release_year"] <= 2022], s[s["release_year"] == 2023]
    search = tune(past)
    model = calibrate(search.best_estimator_, past)
    p = model.predict_proba(future[FEATURES])[:, 1]
    return {"train_years": "2020–2022", "test_year": 2023, "best_C": search.best_params_["clf__C"],
            "roc_auc": round(roc_auc_score(future[TARGET], p), 3),
            "average_precision": round(average_precision_score(future[TARGET], p), 3)}


def learning_curve_table(train, C_value):
    """Train vs CV ROC-AUC for growing training sizes (training data only)."""
    model = make_logreg().set_params(clf__C=C_value)
    sizes, tr, cv = learning_curve(model, train[FEATURES], train[TARGET], cv=cv_splitter(),
                                   scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 8),
                                   n_jobs=-1, shuffle=True, random_state=SEED)
    return pd.DataFrame({"train_size": sizes, "train_auc": tr.mean(1), "train_std": tr.std(1),
                         "cv_auc": cv.mean(1), "cv_std": cv.std(1)})


def random_forest_comparison(train, test):
    """Comparison model for the notebook: shallow, tuned the same way, also calibrated."""
    rf = Pipeline([("prep", make_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1))])
    search = tune(train, rf, {"clf__max_depth": [6, 8, 10], "clf__min_samples_leaf": [20, 50]})
    with warnings.catch_warnings():
        # sigmoid calibration on tree scores can emit harmless overflow warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.calibration")
        model = calibrate(search.best_estimator_, train)
    p_train = model.predict_proba(train[FEATURES])[:, 1]
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    i = search.best_index_
    return {"best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "train_auc": round(roc_auc_score(train[TARGET], p_train), 3),
            "cv_auc_mean": round(search.cv_results_["mean_test_score"][i], 3),
            "cv_auc_std": round(search.cv_results_["std_test_score"][i], 3),
            "test_auc": round(roc_auc_score(test[TARGET], p_test), 3),
            "average_precision": round(average_precision_score(test[TARGET], p_test), 3),
            "brier": round(brier_score_loss(test[TARGET], p_test), 4)}


# ---------------------------------------------------------------- training
def train_and_save(df=None, verbose=True):
    s = load_data(df)
    train, test = split(s)

    # 1) tune C on the training data (5-fold CV)
    search = tune(train)
    best_C = search.best_params_["clf__C"]
    i = search.best_index_
    cv_scores = [search.cv_results_[f"split{k}_test_score"][i] for k in range(5)]

    # 2) calibrate probabilities, still inside the training data
    model = calibrate(search.best_estimator_, train)
    explainer = search.best_estimator_            # refit on all training rows with best C

    # 3) overfitting checks on the training data
    p_train = model.predict_proba(train[FEATURES])[:, 1]
    threshold = float(np.quantile(p_train, 0.9))  # flag the top 10% (chosen without the test set)

    # 4) the test set, used once
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    tm = test_metrics(test[TARGET].to_numpy(), p_test, threshold)
    train_auc = round(roc_auc_score(train[TARGET], p_train), 3)

    x_train = _dense(explainer.named_steps["prep"].transform(train[FEATURES]))
    bundle = {"model": model, "explainer": explainer,
              "feature_names": feature_names(explainer),
              "feature_means": x_train.mean(axis=0), "features": FEATURES}
    metrics = {
        "rows": {"study": int(len(s)), "train": int(len(train)), "test": int(len(test))},
        "features": FEATURES, "best_C": best_C,
        "cv_grid": [{"C": c, "cv_auc": round(m, 4), "cv_std": round(sd, 4),
                     "train_auc": round(t, 4)} for c, m, sd, t in zip(
            C_GRID, search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"],
            search.cv_results_["mean_train_score"])],
        "cv_auc_mean": round(float(np.mean(cv_scores)), 3),
        "cv_auc_std": round(float(np.std(cv_scores)), 3),
        "train_auc": train_auc, "test": tm,
        "overfit_gap": round(train_auc - tm["roc_auc"], 3),
        "overfit_flag": bool(train_auc - tm["roc_auc"] > OVERFIT_GAP),
        "time_check": time_check(s),
        "indie_base_rate_pct": round(100 * s.loc[s["is_indie"] == 1, TARGET].mean(), 2),
    }
    C.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.MODEL_FILE)
    C.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"Best C = {best_C} | train AUC {train_auc} | CV AUC {metrics['cv_auc_mean']} "
              f"± {metrics['cv_auc_std']} | test AUC {tm['roc_auc']} | "
              f"time check (2020–22 → 2023) {metrics['time_check']['roc_auc']}")
        print(f"AP {tm['average_precision']} | Brier {tm['brier']} | top-10% hit rate "
              f"{tm['top10_hit_rate_pct']}% vs base {tm['base_rate_pct']}% | "
              f"overfit flag: {metrics['overfit_flag']}")
        print(f"Saved -> {C.MODEL_FILE}\nSaved -> {C.METRICS_FILE}")
    return bundle, metrics


if __name__ == "__main__":
    # Import through the package so the saved pipeline refers to src.model.split_labels,
    # not __main__.split_labels (which the dashboard could not load).
    from src.model import train_and_save as _train
    _train()
