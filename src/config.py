"""Project paths, study settings and shared constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------- Paths ----------
RAW_FILE = Path(os.environ.get("STEAM_RAW_FILE", ROOT / "data" / "raw" / "games_march2025_full.csv"))
CLEAN_FILE = ROOT / "data" / "processed" / "clean_data.csv"
CLEANING_LOG = ROOT / "data" / "processed" / "cleaning_log.csv"
PIPELINE_DIR = ROOT / "data" / "processed" / "pipeline"   # small files for the Data Pipeline tab
MODEL_FILE = ROOT / "models" / "hit_model.joblib"        # built by python -m src.model
METRICS_FILE = ROOT / "models" / "metrics.json"
FIGURES = ROOT / "reports" / "figures"

DATASET_URL = "https://www.kaggle.com/datasets/artermiloff/steam-games-dataset"
SNAPSHOT_DATE = "2025-03-10"   # latest release date in the file = collection date

# ---------- Study design ----------
STUDY_YEARS = (2020, 2023)     # main analysis cohort
SENSITIVITY_YEAR = 2024        # kept for a robustness check only
HIT_PCT = 0.95                 # hit = top 5% of review count within release year
HIT_PCT_ALT = 0.90             # robustness check: top 10%
MIN_REVIEWS_RATING = 50        # rating % is only trusted from this many reviews
MIN_SAMPLE = 30                # dashboard warns below this group size

# ---------- Columns read from the raw file ----------
RAW_COLUMNS = [
    "appid", "name", "release_date", "price", "required_age", "genres", "categories",
    "tags", "developers", "publishers", "supported_languages", "windows", "mac", "linux",
    "achievements", "dlc_count", "num_reviews_total", "pct_pos_total",
    "average_playtime_forever", "median_playtime_forever", "peak_ccu", "estimated_owners",
]

# Steam "genres" that describe software, not games
SOFTWARE_GENRES = {
    "Utilities", "Design & Illustration", "Animation & Modeling", "Video Production",
    "Audio Production", "Software Training", "Photo Editing", "Web Publishing",
    "Education", "Accounting", "Game Development", "Movie", "Documentary",
    "Episodic", "Short", "Tutorial", "360 Video",
}

# Adult-only content: removed in cleaning (genre, top player tags, name or manual list).
# No popularity exemption: deciding who stays by review count would bias the hit label.
ADULT_GENRES = {"Sexual Content", "Nudity"}
ADULT_TAGS_TOP5 = {"Sexual Content", "Nudity"}    # adult if among a game's top-5 tags
ADULT_TAGS_TOP10 = {"NSFW", "Hentai"}             # adult if among a game's top-10 tags
# Whole words only, except "sex" and "fuck", which also match as word starts ("SexyTails").
# "Sextet" (a group of six) is exempt; "fuck" titles are removed as inappropriate even when
# not sexual.
ADULT_NAME_PATTERN = (
    r"(?<![\w])(?:sex(?!tets?(?![\w]))\w*|fuck\w*|hentai|futa|futanari|nsfw|porn|porno|erotic|erotica|eroge"
    r"|lewd|bdsm|fap|cum|nude|nudes|naked|strip|stripper|milf|milfs|boob|boobs|tits|ecchi"
    r"|harem|step[ -]?sis|step[ -]?sisters?|r18|18\+|lust|lustful|horny|fetish|succubus"
    r"|debauchery|pussy|breasts?|cuckold|seduction)(?![\w+])"
    r"|\U0001F51E"                                # the "no one under 18" symbol
)
# Explicit games that no generic rule catches. "Furry" alone is not a rule because it
# would also remove normal games, so these titles are listed by appid instead.
ADULT_APPIDS = {
    1861970,   # Mister Furry
    1578220,   # Mister Furry: Hot Muscles
    1399930,   # Furry Shades of Gay
    1953540,   # Furry Shades of Gay 3: Still Gayer
}

# Core gameplay genres used for main_genre (Indie / Early Access / Free To Play are
# business labels, so they are kept as separate flags instead)
CORE_GENRES = [
    "Action", "Adventure", "Casual", "Simulation", "Strategy", "RPG",
    "Sports", "Racing", "Massively Multiplayer",
]

# Gameplay sub-genres taken from player tags (much more specific than Steam genres)
SUBGENRE_TAGS = [
    "Puzzle", "Platformer", "Horror", "Visual Novel", "Shooter", "Point & Click",
    "Hidden Object", "Roguelike", "Roguelite", "Bullet Hell", "Survival", "Sandbox",
    "Open World", "Tower Defense", "Card Game", "Deckbuilding", "Walking Simulator",
    "Turn-Based Strategy", "JRPG", "Action RPG", "Metroidvania", "City Builder",
    "Colony Sim", "Farming Sim", "Life Sim", "Fighting", "Rhythm", "Stealth",
    "Management", "Automation", "Souls-like", "Idler", "Racing", "Sports",
    "Party Game", "Battle Royale", "MMORPG", "Dating Sim", "Detective", "Cozy",
]
# Different spellings in Steam tags that mean the same sub-genre
TAG_ALIASES = {
    "Rogue-like": "Roguelike", "Action Roguelike": "Roguelike",
    "Rogue-lite": "Roguelite",
    "2D Platformer": "Platformer", "3D Platformer": "Platformer",
    "Puzzle-Platformer": "Platformer", "Precision Platformer": "Platformer",
    "Psychological Horror": "Horror", "Survival Horror": "Horror",
    "FPS": "Shooter", "Third-Person Shooter": "Shooter", "Shoot 'Em Up": "Shooter",
    "Top-Down Shooter": "Shooter", "Arena Shooter": "Shooter", "Twin Stick Shooter": "Shooter",
    "Deckbuilding": "Deckbuilding", "Card Battler": "Card Game",
    "Clicker": "Idler", "Incremental": "Idler",
    "2D Fighter": "Fighting", "3D Fighter": "Fighting", "Beat 'em up": "Fighting",
    "Farming": "Farming Sim",
}
TAG_TOP_N = 10                 # a game "has" a sub-genre if it is among its top-10 tags

# Well-known games shown as raw-vs-clean examples (last one has num_reviews_total = -1)
PIPELINE_SAMPLE_APPIDS = [1145360, 892970, 1966720, 1794680, 1245620, 739630, 1086940, 331430]

PRICE_BANDS = [-0.01, 0, 4.99, 9.99, 19.99, 10_000]
PRICE_LABELS = ["Free", "Under $5", "$5–10", "$10–20", "$20+"]

# ---------- Visual identity (shared by plots, notebook and dashboard) ----------
COLORS = {
    "bg": "#1e2140",         # page / figure background (retro navy)
    "panel": "#2a2e57",      # cards
    "text": "#f4f1de",
    "muted": "#b8b5d6",
    "grid": "#3a3f6e",
    # Validated together on the dark surface (colour-blind safe, ΔE checked)
    "indie": "#9f7aea",      # purple  -> Indie games
    "other": "#22a884",      # teal    -> Non-Indie games
    "single": "#3b8fe0",     # blue    -> single-series charts (never next to purple)
    "warn": "#e8663d",       # orange  -> "below average" / divisive
    "neutral": "#7c809e",    # gray    -> not significant / low traction
    "accent": "#ffd23f",     # yellow  -> highlights only (titles, key numbers), never a data series
}
QUADRANT_COLORS = {
    "Popular & Loved": COLORS["other"],
    "Popular but Divisive": COLORS["warn"],
    "Hidden Gem": COLORS["indie"],
    "Low Traction": COLORS["neutral"],
}
