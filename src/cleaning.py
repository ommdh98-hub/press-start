"""Cleaning, feature engineering and target definition.

Run from the project root:
    python -m src.cleaning

Reads the untouched raw file, writes:
    data/processed/clean_data.csv   (one row per game, 2020–2024)
    data/processed/cleaning_log.csv (rows before/after every step)
"""
import ast
import json
import re

import numpy as np
import pandas as pd

from src import config as C

TEST_BUILD_PATTERN = re.compile(
    r"\b(?:playtest|beta test|alpha test|open beta|closed beta|public test|test server)\b",
    flags=re.IGNORECASE,
)
MULTIPLAYER_CATEGORIES = {"Multi-player", "Online PvP", "Online Co-op", "Co-op", "PvP",
                          "Shared/Split Screen", "Cross-Platform Multiplayer", "MMO"}


# ---------------------------------------------------------------- helpers
def parse_list(value):
    """"['Action', 'Indie']" -> ['Action', 'Indie'] (empty list if unreadable)."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        return list(parsed.keys())
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def parse_tags(value):
    """"{'Puzzle': 120, 'Indie': 80}" -> ['Puzzle', 'Indie'] sorted by player votes."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        return [k for k, _ in sorted(parsed.items(), key=lambda kv: -kv[1])]
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def subgenres_from_tags(tags):
    """Map a game's top tags to our sub-genre list (handles different spellings)."""
    found = []
    for tag in tags[:C.TAG_TOP_N]:
        tag = C.TAG_ALIASES.get(tag, tag)
        if tag in C.SUBGENRE_TAGS and tag not in found:
            found.append(tag)
    return found


def owners_range(value):
    """'20000 - 50000' -> (20000, 50000)."""
    nums = [int(n) for n in re.findall(r"\d+", str(value))]
    return (nums[0], nums[-1]) if nums else (np.nan, np.nan)


def is_software(df):
    """Apps whose only Steam genres are software genres."""
    genre_lists = df["genres"].apply(parse_list)
    return genre_lists.apply(lambda g: len(g) > 0 and set(g) <= C.SOFTWARE_GENRES)


def is_adult(df):
    """Adult genre, adult top-5/top-10 tags, explicit name or manual appid list."""
    adult_genre = df["genres"].apply(parse_list).apply(lambda g: bool(C.ADULT_GENRES & set(g)))
    tag_lists = df["tags"].apply(parse_tags)
    adult_tag = tag_lists.apply(lambda t: bool(C.ADULT_TAGS_TOP5 & set(t[:5]))
                                or bool(C.ADULT_TAGS_TOP10 & set(t[:10])))
    adult_name = df["name"].astype(str).str.contains(C.ADULT_NAME_PATTERN, case=False, regex=True)
    return adult_genre | adult_tag | adult_name | df["appid"].isin(C.ADULT_APPIDS)


def name_key(names):
    return names.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def is_duplicate_edition(df):
    """Later pages sharing an identical review pool with an earlier one (df sorted by appid)."""
    has_reviews = df["num_reviews_total"] >= 10
    same_name = df.assign(_n=name_key(df["name"])).duplicated(
        ["_n", "num_reviews_total", "pct_pos_total"])
    same_dev = df.duplicated(["developers", "num_reviews_total", "pct_pos_total"]) & \
        (df["num_reviews_total"] >= 200)
    return has_reviews & (same_name | same_dev)


class CleaningLog:
    """Records the row count before and after every cleaning step."""

    def __init__(self, df):
        self.rows = []
        self.current = len(df)
        self.add("Raw file loaded", df, "Original Kaggle file (March 2025 snapshot)")

    def add(self, step, df, reason):
        self.rows.append({"step": step, "rows_before": self.current, "rows_after": len(df),
                          "removed": self.current - len(df), "reason": reason})
        self.current = len(df)

    def table(self):
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- pipeline
def load_raw(path=C.RAW_FILE, columns=None):
    """Read only the columns we need; the raw file itself is never modified."""
    return pd.read_csv(path, usecols=columns or C.RAW_COLUMNS, low_memory=False)


def clean(raw, verbose=True):
    """Full pipeline. Returns (clean_df, log_df)."""
    df = raw.copy()
    log = CleaningLog(df)

    # 1) Test builds (Playtest / Beta) are not real releases and some of them
    #    show the review count of the main game.
    df = df[~df["name"].astype(str).str.contains(TEST_BUILD_PATTERN)]
    log.add("Remove test builds", df, "Playtest / beta builds are not released games")

    # 2) Software (e.g. video editors) is not a game
    df = df[~is_software(df)]
    log.add("Remove software", df, "Only software genres (Utilities, Video Production…)")

    # 2b) Adult-only content is a separate market, not a target for our persona
    df = df[~is_adult(df)]
    log.add("Remove adult content", df,
            "Adult genre, adult top-5/top-10 tags, explicit name or manual list")

    # 3) Exact duplicate ids
    df = df.drop_duplicates(subset="appid")
    log.add("Remove duplicate appid", df, "Same Steam id listed twice")

    # 4) Duplicate editions: several store pages sharing one review pool
    #    (e.g. Standard + Gold edition). Keep the first (lowest appid).
    df = df.sort_values("appid")
    df = df[~is_duplicate_edition(df)]
    log.add("Remove duplicate editions", df,
            "Editions sharing an identical review pool (same name or same developer)")

    # 5) Dates
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    log.add("Remove invalid dates", df, "Release date could not be parsed")

    # ------------------------------------------------ feature engineering
    df = add_features(df)

    # 6) Study scope: games released 2020–2024 (2024 kept only for robustness)
    first, last = C.STUDY_YEARS
    df = df[df["release_year"].between(first, C.SENSITIVITY_YEAR)].copy()
    log.add("Keep releases 2020–2024", df,
            "Recent market; 2024 games are used only as a robustness check")

    # ------------------------------------------------ target (hit) definition
    df = add_targets(df)

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    log_df = log.table()
    if verbose:
        print(log_df.to_string(index=False))
        print(f"\nFinal: {len(df):,} rows × {df.shape[1]} columns")
    return df, log_df


def add_features(df):
    df = df.copy()
    genres = df["genres"].apply(parse_list)
    tags = df["tags"].apply(parse_tags)
    categories = df["categories"].apply(parse_list)
    languages = df["supported_languages"].apply(parse_list)

    # Release timing
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_quarter"] = df["release_date"].dt.to_period("Q").astype(str)
    df["days_on_market"] = (pd.Timestamp(C.SNAPSHOT_DATE) - df["release_date"]).dt.days

    # Reviews: Steam gives no review summary below 10 reviews -> the file stores -1.
    # That is "fewer than 10 reviews", not missing data.
    df["few_reviews"] = df["num_reviews_total"] < 10
    df["review_count"] = df["num_reviews_total"].clip(lower=0)
    df["rating_pct"] = df["pct_pos_total"].where(~df["few_reviews"])

    # Business labels
    df["is_indie"] = genres.apply(lambda g: "Indie" in g)
    df["is_early_access"] = genres.apply(lambda g: "Early Access" in g)
    df["is_free"] = df["price"] == 0
    df["price_band"] = pd.cut(df["price"], bins=C.PRICE_BANDS, labels=C.PRICE_LABELS)

    # Genres
    # A game can have several genres (Steam lists them alphabetically, so there is
    # no "main" one) -> keep all core genres as a multi-label text column
    df["genres_core"] = genres.apply(lambda g: "|".join(x for x in g if x in C.CORE_GENRES))
    df["n_genres"] = genres.apply(lambda g: len([x for x in g if x in C.CORE_GENRES]))
    df["has_tags"] = tags.apply(len) > 0
    df["subgenres"] = tags.apply(lambda t: "|".join(subgenres_from_tags(t)))

    # Product features
    df["n_languages"] = languages.apply(len)
    df["arabic_support"] = languages.apply(lambda l: "Arabic" in l)
    df["n_platforms"] = df[["windows", "mac", "linux"]].astype(bool).sum(axis=1)
    df["has_achievements"] = df["achievements"] > 0
    df["is_multiplayer"] = categories.apply(lambda c: bool(MULTIPLAYER_CATEGORIES & set(c)))
    df["has_dlc"] = df["dlc_count"] > 0
    df["playtime_hours"] = (df["average_playtime_forever"] / 60).round(1)

    # Studio
    devs = df["developers"].apply(parse_list)
    pubs = df["publishers"].apply(parse_list)
    df["developer"] = devs.apply(lambda d: d[0] if d else "Unknown")
    df["self_published"] = [bool(d) and set(d) == set(p) for d, p in zip(devs, pubs)]
    # Catalogue size counted over the whole snapshot (may include later games)
    df["developer_titles"] = df.groupby("developer")["appid"].transform("count")
    df.loc[df["developer"] == "Unknown", "developer_titles"] = np.nan

    # Owners are SteamSpy ranges (estimates, not sales)
    bounds = df["estimated_owners"].apply(owners_range)
    df["owners_low"] = bounds.str[0]
    df["owners_high"] = bounds.str[1]
    return df


def add_targets(df):
    """Hit = top 5% of review count among games released in the same year."""
    df = df.copy()
    by_year = df.groupby("release_year")["review_count"]
    by_quarter = df.groupby("release_quarter")["review_count"]

    df["visibility_pctile"] = (by_year.rank(pct=True, method="max") * 100).round(1)
    df["is_hit"] = df["review_count"] >= by_year.transform(lambda s: s.quantile(C.HIT_PCT))
    df["is_hit_top10"] = df["review_count"] >= by_year.transform(lambda s: s.quantile(C.HIT_PCT_ALT))
    df["is_hit_by_quarter"] = df["review_count"] >= by_quarter.transform(
        lambda s: s.quantile(C.HIT_PCT))
    # A game with < 10 reviews can never be a hit (protects tiny groups)
    for col in ["is_hit", "is_hit_top10", "is_hit_by_quarter"]:
        df.loc[df["few_reviews"], col] = False

    first, last = C.STUDY_YEARS
    df["cohort"] = np.where(df["release_year"].between(first, last), "Study 2020–2023",
                            "Robustness 2024")

    # Popularity vs satisfaction quadrant (only with enough reviews to trust the rating)
    rated = df["review_count"] >= C.MIN_REVIEWS_RATING
    loved = df["rating_pct"] >= 80          # Steam's "Very Positive" starts at 80%
    df["quadrant"] = np.select(
        [rated & df["is_hit"] & loved, rated & df["is_hit"] & ~loved,
         rated & ~df["is_hit"] & loved, rated & ~df["is_hit"] & ~loved],
        ["Popular & Loved", "Popular but Divisive", "Hidden Gem", "Low Traction"],
        default="Not enough reviews",
    )
    return df


OUTPUT_COLUMNS = [
    # identity
    "appid", "name", "developer", "release_date", "release_year", "release_month",
    "release_quarter", "cohort", "days_on_market",
    # business labels
    "is_indie", "is_early_access", "is_free", "price", "price_band", "self_published",
    "developer_titles",
    # content
    "genres_core", "n_genres", "subgenres", "has_tags", "is_multiplayer", "n_languages",
    "arabic_support", "n_platforms", "has_achievements", "achievements", "has_dlc",
    "required_age",
    # outcomes
    "review_count", "few_reviews", "rating_pct", "playtime_hours", "peak_ccu",
    "owners_low", "owners_high",
    # targets
    "visibility_pctile", "is_hit", "is_hit_top10", "is_hit_by_quarter", "quadrant",
]


def save(df, log_df):
    C.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_FILE, index=False)
    log_df.to_csv(C.CLEANING_LOG, index=False)
    print(f"Saved -> {C.CLEAN_FILE}")
    print(f"Saved -> {C.CLEANING_LOG}")


# ---------------------------------------------------------------- pipeline artefacts
# Small files that let the dashboard show the raw data and every cleaning step without
# loading the 470 MB raw file. Examples are always taken from games kept in the clean data.
SAMPLE_COLUMNS = ["appid", "name", "release_date", "price", "genres", "tags", "num_reviews_total",
                  "pct_pos_total", "positive", "negative", "estimated_owners",
                  "supported_languages"]
CLEAN_SAMPLE_COLUMNS = ["appid", "name", "is_indie", "few_reviews", "review_count", "rating_pct",
                        "owners_low", "owners_high", "genres_core", "subgenres", "n_languages",
                        "price_band", "is_hit", "quadrant"]


def _is_blank(s):
    """Missing value, or an empty text / list / dict as the raw file stores them."""
    return s.isna() | s.astype(str).str.strip().isin(["", "[]", "{}"])


def _short(value, n):
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= n else text[:n - 1] + "…"


def profile_raw(path, prefer_appids, keep_appids, chunksize=10_000):
    """dtype, missing %, unique values and an example for every raw column (chunked read).

    Examples come from `prefer_appids` (well-known games) first, then any kept game.
    """
    stats, n_rows = {}, 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        n_rows += len(chunk)
        pref = chunk[chunk["appid"].isin(prefer_appids)]
        kept = chunk[chunk["appid"].isin(keep_appids)]
        for col in chunk.columns:
            st_ = stats.setdefault(col, {"dtypes": set(), "missing": 0, "hashes": set(),
                                         "example": None, "example_rank": None})
            s = chunk[col]
            st_["dtypes"].add(str(s.dtype))
            blank = _is_blank(s)
            st_["missing"] += int(blank.sum())
            st_["hashes"].update(pd.util.hash_pandas_object(s[~blank], index=False).tolist())
            for rank, src in ((0, pref), (1, kept)):
                v = src[col][~_is_blank(src[col])]
                if len(v) and (st_["example_rank"] is None or rank < st_["example_rank"]):
                    if rank == 0:   # keep the preferred games' order (first listed wins)
                        order = {a: i for i, a in enumerate(prefer_appids)}
                        v = v.loc[src.loc[v.index, "appid"].map(order).sort_values().index]
                    st_["example"], st_["example_rank"] = v.iloc[0], rank
    rows = []
    for col, st_ in stats.items():
        dtypes = st_["dtypes"]
        dtype = dtypes.pop() if len(dtypes) == 1 else \
            ("float64" if dtypes <= {"int64", "float64"} else "object")
        rows.append({"column": col, "dtype": dtype,
                     "missing_pct": round(100 * st_["missing"] / n_rows, 1),
                     "unique_values": len(st_["hashes"]),
                     "example": "" if st_["example"] is None else _short(st_["example"], 80)})
    return pd.DataFrame(rows)


def quality_examples(raw):
    """Evidence for the four data-quality decisions (long format: finding, label, value, detail)."""
    out = []
    nrt = raw["num_reviews_total"]
    out += [("sentinel", "Smallest real review count", int(nrt[nrt > 0].min()), ""),
            ("sentinel", "Rows with -1", int((nrt == -1).sum()), ""),
            ("sentinel", "Share of rows with -1 (%)", round(100 * (nrt == -1).mean(), 1), "")]

    pn = raw["positive"] + raw["negative"]
    both = (pn > 0) & (nrt > 0)
    year = pd.to_datetime(raw["release_date"], errors="coerce").dt.year
    cover = (pn > 0).groupby(year).mean() * 100
    out += [("sources", "Games with both counts", int(both.sum()), ""),
            ("sources", "Identical counts (%)", round(100 * (pn[both] == nrt[both]).mean(), 1), ""),
            ("sources", "SteamSpy coverage 2020 (%)", round(cover.get(2020, np.nan), 1), ""),
            ("sources", "SteamSpy coverage 2023 (%)", round(cover.get(2023, np.nan), 1), "")]

    ok = ~is_adult(raw) & ~is_software(raw)
    tests = raw[ok & raw["name"].astype(str).str.contains(TEST_BUILD_PATTERN)]
    for r in tests.nlargest(3, "num_reviews_total").itertuples():
        out.append(("playtests", r.name, int(r.num_reviews_total), ""))

    # Duplicate editions exactly as step 4 sees them, paired with the page that is kept
    pool = raw[ok & ~raw["name"].astype(str).str.contains(TEST_BUILD_PATTERN)]
    pool = pool.drop_duplicates(subset="appid").sort_values("appid")
    dup = is_duplicate_edition(pool)
    keyed = pool.assign(key=name_key(pool["name"]))
    kept = keyed[~dup]
    by_name = kept.drop_duplicates(["key", "num_reviews_total", "pct_pos_total"]).set_index(
        ["key", "num_reviews_total", "pct_pos_total"])["name"]
    by_dev = kept.drop_duplicates(["developers", "num_reviews_total", "pct_pos_total"]).set_index(
        ["developers", "num_reviews_total", "pct_pos_total"])["name"]
    removed = keyed[dup].copy()
    removed["kept_as"] = [
        by_name.get((r.key, r.num_reviews_total, r.pct_pos_total),
                    by_dev.get((r.developers, r.num_reviews_total, r.pct_pos_total), r.name))
        for r in removed.itertuples()]
    # One example per kept game: its shared review count and how many extra pages were removed
    groups = removed.groupby("kept_as").agg(reviews=("num_reviews_total", "max"),
                                            extra_pages=("appid", "count"))
    for kept_name, g in groups.nlargest(3, "reviews").iterrows():
        out.append(("duplicates", kept_name, int(g["reviews"]), int(g["extra_pages"])))
    return pd.DataFrame(out, columns=["finding", "label", "value", "detail"])


def missing_before_after(raw, clean_df):
    """Missing % of key fields in the raw file vs the same fields in the clean data."""
    nrt, pct = raw["num_reviews_total"], raw["pct_pos_total"]
    pairs = [
        ("Review count", nrt.isna() | (nrt < 0), clean_df["review_count"].isna()),
        ("Rating %", pct.isna() | (pct < 0) | (nrt < 10), clean_df["rating_pct"].isna()),
        ("Release date", pd.to_datetime(raw["release_date"], errors="coerce").isna(),
         clean_df["release_date"].isna()),
        ("Price", raw["price"].isna(), clean_df["price"].isna()),
        ("Developer", _is_blank(raw["developers"]), clean_df["developer"].eq("Unknown")),
        ("Player tags", _is_blank(raw["tags"]), ~clean_df["has_tags"].astype(bool)),
        ("Languages", _is_blank(raw["supported_languages"]), clean_df["n_languages"].eq(0)),
        ("Estimated owners", _is_blank(raw["estimated_owners"]), clean_df["owners_low"].isna()),
    ]
    return pd.DataFrame([{"field": f, "raw_missing_pct": round(100 * a.mean(), 1),
                          "clean_missing_pct": round(100 * b.mean(), 1)} for f, a, b in pairs])


def save_pipeline(clean_df, raw_path=C.RAW_FILE):
    """Write the small pipeline files used by the dashboard's Data Pipeline tab."""
    out = C.PIPELINE_DIR
    out.mkdir(parents=True, exist_ok=True)
    keep = set(clean_df["appid"])
    sample_ids = [a for a in C.PIPELINE_SAMPLE_APPIDS if a in keep]

    raw = pd.read_csv(raw_path, usecols=sorted(set(C.RAW_COLUMNS + SAMPLE_COLUMNS)),
                      low_memory=False)
    profile_raw(raw_path, sample_ids, keep).to_csv(out / "raw_profile.csv", index=False)

    sample = raw[raw["appid"].isin(sample_ids)][SAMPLE_COLUMNS].copy()
    sample["tags"] = sample["tags"].map(lambda t: _short(t, 120))
    order = {a: i for i, a in enumerate(sample_ids)}
    sample = sample.sort_values("appid", key=lambda s: s.map(order))
    sample.to_csv(out / "raw_sample.csv", index=False)
    clean_sample = clean_df[clean_df["appid"].isin(sample_ids)][CLEAN_SAMPLE_COLUMNS]
    clean_sample.sort_values("appid", key=lambda s: s.map(order)).to_csv(
        out / "clean_sample.csv", index=False)

    quality_examples(raw).to_csv(out / "quality_examples.csv", index=False)
    missing_before_after(raw, clean_df).to_csv(out / "missing_before_after.csv", index=False)

    summary = {"raw_rows": int(len(raw)),
               "raw_columns": int(len(pd.read_csv(raw_path, nrows=0).columns)),
               "raw_size_mb": round(raw_path.stat().st_size / 1e6, 1),
               "clean_rows": int(len(clean_df)), "clean_columns": int(clean_df.shape[1]),
               "clean_size_mb": round(C.CLEAN_FILE.stat().st_size / 1e6, 1)}
    (out / "raw_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved pipeline files -> {out}")


def load_clean(path=C.CLEAN_FILE):
    """Load the processed file with the right types (used by notebook + app)."""
    df = pd.read_csv(path, parse_dates=["release_date"])
    df["price_band"] = pd.Categorical(df["price_band"], categories=C.PRICE_LABELS, ordered=True)
    df["subgenres"] = df["subgenres"].fillna("")
    df["genres_core"] = df["genres_core"].fillna("")
    return df


if __name__ == "__main__":
    clean_df, log_table = clean(load_raw())
    save(clean_df, log_table)
    save_pipeline(clean_df)
