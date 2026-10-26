"""Shared calculations used by the notebook and the dashboard.

Every function takes the cleaned DataFrame and returns a small table,
so the numbers shown in the notebook and in the app always match.
"""
import numpy as np
import pandas as pd

from src import config as C

HIT_DEFINITIONS = {
    "is_hit": "Top 5% within release year (main)",
    "is_hit_top10": "Top 10% within release year",
    "is_hit_by_quarter": "Top 5% within release quarter",
}


# ---------------------------------------------------------------- basics
def spearman(data, other=None):
    """Spearman correlation without SciPy (= Pearson correlation of the ranks)."""
    if other is not None:
        return data.rank().corr(other.rank())
    return data.rank().corr()

def study(df):
    """Main analysis cohort: games released 2020–2023."""
    first, last = C.STUDY_YEARS
    return df[df["release_year"].between(first, last)]


def wilson_ci(hits, n, z=1.96):
    """95% confidence interval for a proportion (works well for small groups)."""
    hits, n = np.asarray(hits, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = hits / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(df, by, target="is_hit", min_n=1):
    """Hit rate (%) per group with sample size and 95% CI."""
    t = df.groupby(by, observed=True)[target].agg(games="size", hits="sum").reset_index()
    t = t[t["games"] >= min_n].copy()
    lo, hi = wilson_ci(t["hits"], t["games"])
    t["hit_rate"] = t["hits"] / t["games"] * 100
    t["ci_low"], t["ci_high"] = lo * 100, hi * 100
    return t


def explode_labels(df, col):
    """'Puzzle|Horror' -> one row per label (a game can count in several groups)."""
    out = df.assign(**{col: df[col].fillna("").str.split("|")}).explode(col)
    return out[out[col] != ""]


# ---------------------------------------------------------------- key questions
def overview_numbers(df):
    s = study(df)
    return {
        "games_all": len(df),
        "games_study": len(s),
        "indie_share": s["is_indie"].mean() * 100,
        "hit_rate_all": s["is_hit"].mean() * 100,
        "hits": int(s["is_hit"].sum()),
        "few_reviews_share": s["few_reviews"].mean() * 100,
        "median_reviews": s["review_count"].median(),
        "hit_threshold_by_year": s.groupby("release_year")["review_count"]
                                  .quantile(C.HIT_PCT).round(0).to_dict(),
    }


def indie_gap(df):
    """Hit rate for Indie vs non-Indie under every hit definition (robustness)."""
    rows = []
    for col, label in HIT_DEFINITIONS.items():
        t = rate_table(study(df), "is_indie", target=col)
        t["definition"] = label
        rows.append(t)
    # Same comparison on the 2024 cohort
    t = rate_table(df[df["release_year"] == C.SENSITIVITY_YEAR], "is_indie")
    t["definition"] = "Top 5% — 2024 releases only"
    rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out["group"] = out["is_indie"].map({True: "Indie", False: "Non-Indie"})
    return out[["definition", "group", "games", "hits", "hit_rate", "ci_low", "ci_high"]]


def indie_share_of_hits(df):
    s = study(df)
    return {"share_of_games": s["is_indie"].mean() * 100,
            "share_of_hits": s.loc[s["is_hit"], "is_indie"].mean() * 100}


def opportunity_table(df, col="subgenres", indie_only=True, min_n=80):
    """Competition (number of games) vs hit rate for every genre / sub-genre."""
    s = study(df)
    if indie_only:
        s = s[s["is_indie"]]
    if col == "subgenres":
        s = s[s["has_tags"]]          # tags are missing for 21% of games
    t = rate_table(explode_labels(s, col), col, min_n=min_n)
    base = s["is_hit"].mean() * 100
    t["vs_average"] = t["hit_rate"] - base
    t["signal"] = np.select(
        [t["ci_low"] > base, t["ci_high"] < base], ["Above average", "Below average"],
        default="Not clearly different")
    return t.sort_values("hit_rate", ascending=False).rename(columns={col: "label"}), base


def feature_effects(df, indie_only=True):
    """Hit rate by simple product choices (associations, not causes)."""
    s = study(df)
    if indie_only:
        s = s[s["is_indie"]]
    s = s.assign(
        languages=pd.cut(s["n_languages"], [0, 1, 4, 9, 200],
                         labels=["1", "2–4", "5–9", "10+"]),
        platforms=s["n_platforms"].map({1: "Windows only", 2: "2 platforms", 3: "3 platforms"}),
        mode=s["is_multiplayer"].map({True: "Has multiplayer", False: "Single-player only"}),
        access=s["is_early_access"].map({True: "Early Access", False: "Full release"}),
        arabic=s["arabic_support"].map({True: "Arabic supported", False: "No Arabic"}),
        achievements=s["has_achievements"].map({True: "Has achievements", False: "None"}),
    )
    factors = {"Price": "price_band", "Languages": "languages", "Platforms": "platforms",
               "Mode": "mode", "Release": "access", "Arabic": "arabic",
               "Achievements": "achievements"}
    rows = []
    for name, col in factors.items():
        t = rate_table(s, col).rename(columns={col: "level"})
        t["factor"] = name
        t["level"] = t["level"].astype(str)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def quadrant_table(df):
    s = study(df)
    s = s[s["quadrant"] != "Not enough reviews"]
    t = s.groupby(["quadrant", "is_indie"]).size().unstack(fill_value=0)
    t.columns = ["Non-Indie", "Indie"]
    t["Indie share %"] = t["Indie"] / t.sum(axis=1) * 100
    return t


def correlation_matrix(df):
    """Spearman correlation. Review count is left out on purpose: the hit label is
    built from it, so their correlation would be artificial."""
    cols = {"price": "Price", "n_languages": "Languages", "n_platforms": "Platforms",
            "achievements": "Achievements", "n_genres": "Genres", "is_multiplayer": "Multiplayer",
            "is_indie": "Indie", "rating_pct": "Rating %",
            "playtime_hours": "Playtime", "is_hit": "Hit"}
    s = study(df)[list(cols)].astype(float).rename(columns=cols)
    return spearman(s)


# ---------------------------------------------------------------- Hit Lab
def similar_games(df, genre=None, subgenre=None, price_band=None, indie_only=True,
                  multiplayer=None):
    """Historical look-alikes for the Hit Lab (descriptive, not a prediction)."""
    s = study(df)
    if indie_only:
        s = s[s["is_indie"]]
    if genre:
        s = s[s["genres_core"].str.split("|").apply(lambda g: genre in g)]
    if subgenre:
        s = s[s["subgenres"].str.split("|").apply(lambda g: subgenre in g)]
    if price_band:
        s = s[s["price_band"].astype(str) == price_band]
    if multiplayer is not None:
        s = s[s["is_multiplayer"] == multiplayer]
    n, hits = len(s), int(s["is_hit"].sum())
    lo, hi = wilson_ci(hits, n) if n else (np.nan, np.nan)
    return s, {"games": n, "hits": hits,
               "hit_rate": hits / n * 100 if n else np.nan,
               "ci_low": float(lo) * 100 if n else np.nan,
               "ci_high": float(hi) * 100 if n else np.nan,
               "enough_data": n >= C.MIN_SAMPLE}
