"""Shared pieces for every dashboard page: style, data, model and small UI helpers."""
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import analysis as A  # noqa: E402
from src import config as C  # noqa: E402
from src import model as M  # noqa: E402
from src.cleaning import load_clean  # noqa: E402

K = C.COLORS
INK = "#0b0d1f"                       # pixel outline + hard shadow
PIXEL = "'Press Start 2P', monospace"
BOX = f"border: 3px solid {INK}; box-shadow: 4px 4px 0 {INK};"

# Pages (paths relative to app/app.py) in menu order: (path, menu label)
PAGES = [("views/home.py", "HOME"), ("views/story.py", "STORY MODE"),
         ("views/hit_lab.py", "HIT LAB"), ("views/arcade.py", "ARCADE")]
# Widget values that must survive a page switch (Streamlit drops values of unseen widgets)
PERSIST_PREFIXES = ("flt_", "pred_")


def inject_css():
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
  *, *::before, *::after {{border-radius: 0 !important;}}
  [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
  [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child * {{border-radius: 50% !important;}}

  /* page = arcade cabinet: faint pixel grid around one framed "game screen" */
  [data-testid="stMain"] {{background-color: {K['bg']};
      background-image: linear-gradient(rgba(244,241,222,.035) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(244,241,222,.035) 1px, transparent 1px);
      background-size: 6px 6px;}}
  [data-testid="stHeader"] {{background: transparent;}}
  [data-testid="stMainBlockContainer"] {{background: {K['bg']}; border: 4px solid {INK};
      box-shadow: 8px 8px 0 {INK}; padding: 22px 24px 28px !important; max-width: 1400px;
      width: calc(100% - 64px); margin: 64px auto 48px;}}

  /* pixel font: title, section headers, tab labels, metric values only */
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
  [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
  [data-testid="stHeading"] h3, [data-testid="stHeading"] h4 {{
      font-family: {PIXEL}; font-weight: 400; line-height: 1.6; letter-spacing: 0;}}
  [data-testid="stMarkdownContainer"] h2, [data-testid="stHeading"] h2 {{font-size: 1rem;}}
  [data-testid="stMarkdownContainer"] h3, [data-testid="stHeading"] h3 {{font-size: 0.9rem;}}
  [data-testid="stMarkdownContainer"] h4, [data-testid="stHeading"] h4 {{font-size: 0.8rem;}}

  /* cards */
  [data-testid="stMetric"] {{background: {K['panel']}; {BOX} padding: 14px 18px;}}
  [data-testid="stMetricLabel"] p {{color: {K['muted']}; font-size: 0.85rem;}}
  [data-testid="stMetricValue"] {{font-family: {PIXEL}; font-size: 1rem; color: {K['accent']};
      padding-top: 6px;}}
  [data-testid="stMetricValue"] div {{font-family: {PIXEL};}}
  .hero {{background: {K['panel']}; {BOX} padding: 22px 26px; margin-bottom: 22px;}}
  .hero h1 {{margin: 0 0 14px 0; font-size: 1.25rem; color: {K['accent']};}}
  .hero.title {{text-align: center; padding: 34px 26px 30px;}}
  .hero.title h1 {{font-size: 2.2rem; margin-bottom: 18px;
      animation: blink 1.2s steps(1, end) infinite;}}
  @keyframes blink {{50% {{opacity: 0.25;}}}}
  @media (prefers-reduced-motion: reduce) {{.hero.title h1 {{animation: none;}}}}
  .hero .tagline {{color: {K['text']}; font-size: 1.2rem; font-weight: 600; margin: 0;}}
  .insight {{background: {K['panel']}; {BOX} border-left: 8px solid {K['indie']};
      padding: 12px 16px; margin: 10px 0;}}
  .caveat {{color: {K['muted']}; font-size: 0.88rem;}}
  [data-testid="stExpander"] details {{{BOX}}}
  [data-testid="stHeaderActionElements"] {{display: none;}}

  /* every section is a card (st.container(key="card_...")); charts stay transparent inside */
  [class*="st-key-card_"], [class*="st-key-mode_"] {{background: {K['panel']}; {BOX}
      padding: 16px 18px; box-sizing: border-box;}}
  [class*="st-key-card_"] [data-testid="stMetric"] {{background: {K['bg']};}}
  [class*="st-key-card_"] [data-testid="stMarkdownContainer"] h4 {{margin-top: 0;}}

  /* cards in the same row end on the same line */
  [data-testid="stHorizontalBlock"] {{align-items: stretch !important;}}
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{height: 100%;}}
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :only-child {{flex: 1 1 auto;}}
  [data-testid="stColumn"] [class*="st-key-card_"],
  [data-testid="stColumn"] [class*="st-key-mode_"],
  [data-testid="stColumn"] [data-testid="stElementContainer"]:only-child > [data-testid="stMetric"] {{
      height: 100%;}}
  [data-testid="stColumn"] [data-testid="stElementContainer"]:only-child:has(> [data-testid="stMetric"]) {{
      height: 100%;}}

  /* key insights = game dialog box */
  [class*="st-key-msgbox_"] {{background: #23264a; border: 3px solid {K['accent']};
      box-shadow: 4px 4px 0 {INK}; padding: 16px 20px 12px; box-sizing: border-box;}}
  /* Hit Predictor game layer */
  .rankrow {{display: flex; gap: 18px; align-items: center; margin: 6px 0 12px;}}
  .rank-label {{font-family: {PIXEL}; font-size: 0.6rem; color: {K['muted']}; margin-bottom: 6px;}}
  .rank {{font-family: {PIXEL}; font-size: 2.6rem; width: 92px; height: 92px; display: flex;
      align-items: center; justify-content: center; border: 4px solid; background: {K['bg']};
      box-shadow: 4px 4px 0 {INK};}}
  .better {{font-family: {PIXEL}; font-size: 0.75rem; line-height: 1.7; color: {K['text']};
      margin-bottom: 6px;}}
  .chance-small {{color: {K['muted']}; font-size: 0.95rem;}}
  .pu {{margin: 0 0 6px 0;}}
  .pu-tag {{font-family: {PIXEL}; font-size: 0.6rem; color: {INK}; background: {K['accent']};
      padding: 3px 6px; margin-right: 8px;}}
  .moved p {{margin: 0 0 4px 0;}}
  .msg-label {{font-family: {PIXEL}; font-size: 0.75rem; color: {K['accent']}; margin-bottom: 4px;}}
  .big {{font-family: {PIXEL}; font-size: 0.95rem; color: {K['accent']}; margin: 2px 0 10px 0;}}

  /* tabs = game buttons; the active one is pressed in */
  .stTabs [data-baseweb="tab-list"] {{gap: 6px; padding-bottom: 8px; flex-wrap: wrap;}}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}
  .stTabs [data-baseweb="tab"] {{background: {K['panel']}; {BOX} padding: 10px 6px;
      height: auto; transition: transform .05s;}}
  .stTabs [data-baseweb="tab"] p {{font-family: {PIXEL}; font-size: 0.6rem; color: {K['text']};}}
  .stTabs [data-baseweb="tab"]:hover {{background: {K['grid']};}}
  .stTabs [aria-selected="true"] {{background: {K['indie']}; transform: translate(4px, 4px);
      box-shadow: inset 3px 3px 0 rgba(0,0,0,.35);}}
  .stTabs [aria-selected="true"] p {{color: {INK};}}

  /* top navigation bar + mode buttons = the same pixel buttons; current page pressed in */
  .st-key-topnav {{margin-bottom: 10px;}}
  .st-key-topnav [data-testid="stPageLink"], [class*="st-key-mode_"] [data-testid="stPageLink"],
  .st-key-topnav [data-testid="stElementContainer"],
  [class*="st-key-mode_"] [data-testid="stElementContainer"] {{width: 100% !important;}}
  .st-key-topnav a, [class*="st-key-mode_"] a {{{BOX} background: {K['panel']}; display: flex;
      width: 100%; box-sizing: border-box; justify-content: center; padding: 12px 8px;
      transition: transform .05s;}}
  .st-key-topnav a *, [class*="st-key-mode_"] a * {{font-family: {PIXEL}; font-size: 0.66rem;
      color: {K['text']}; line-height: 1.6;}}
  .st-key-topnav a:hover, [class*="st-key-mode_"] a:hover {{background: {K['grid']};}}
  .st-key-topnav a:active, [class*="st-key-mode_"] a:active {{transform: translate(4px, 4px);
      box-shadow: none;}}
  [class*="st-key-navcur_"] a, [class*="st-key-navcur_"] a:hover {{background: {K['indie']};
      transform: translate(4px, 4px); box-shadow: inset 3px 3px 0 rgba(0,0,0,.35);}}
  [class*="st-key-navcur_"] a * {{color: {INK};}}

  /* arcade: the game frame is centred in its card, the note sits below it */
  .st-key-card_arcade {{padding: 22px 22px 16px;}}
  .st-key-card_arcade iframe {{display: block; max-width: 1010px; margin: 0 auto;}}
  .game-note {{text-align: center; margin: 14px 0 0;}}

  /* home: three mode cards */
  /* each card is an equal-height flex column; the button is pushed to the bottom */
  [class*="st-key-mode_"] {{text-align: center; padding: 26px 22px 22px; display: flex;
      flex-direction: column; container-type: inline-size;}}
  [class*="st-key-mode_"] > [data-testid="stElementContainer"]:last-child {{margin-top: auto;}}
  [class*="st-key-mode_"] a {{background: {K['accent']}; padding: 18px 8px; margin-top: 6px;}}
  /* pixel texts shrink with the card (cqi = 1% of its width) so each fits on one line; below
     a readable minimum they wrap instead (very narrow windows only) */
  [class*="st-key-mode_"] a * {{color: {INK}; font-size: clamp(0.55rem, 6.6cqi, 0.85rem);}}
  [class*="st-key-mode_"] a:hover {{background: {K['text']};}}
  .picon {{display: block; width: 64px; height: 64px; margin: 4px auto 20px;}}
  .mode-head {{min-height: 5.6rem;}}
  .mode-title {{font-family: {PIXEL}; font-size: clamp(0.7rem, 8.6cqi, 1.05rem); line-height: 1.6;
      color: {K['accent']}; margin: 0 0 14px;}}
  .mode-tag {{font-family: {PIXEL}; font-size: clamp(0.45rem, 4.4cqi, 0.62rem); line-height: 1.7;
      color: {K['text']};}}
  .mode-desc {{color: {K['muted']}; font-size: 0.95rem; line-height: 1.6; margin: 0 0 16px;}}

  /* buttons */
  .stButton button, .stDownloadButton button {{{BOX} background: {K['panel']};}}
  .stButton button:active, .stDownloadButton button:active {{transform: translate(4px, 4px);
      box-shadow: none;}}

  /* Hit Lab progress bar = segmented health bar */
  [data-testid="stProgress"] [role="progressbar"] > div > div {{height: 22px; background: {INK};
      border: 3px solid {INK}; box-shadow: 4px 4px 0 {INK};}}
  [data-testid="stProgress"] [role="progressbar"] > div > div > div {{height: 100%;
      background: repeating-linear-gradient(90deg, {K['accent']} 0 14px, {INK} 14px 18px);}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ UI helpers
# Older Streamlit (<=1.50, last on Python 3.9) has no `width` here and
# silently swallows it into deprecated **kwargs, so check explicitly.
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig):
    """Full-width chart on both newer and older Streamlit versions."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


NICE = {"name": "Game", "release_year": "Year", "release_date": "Released", "price": "Price ($)",
        "review_count": "Reviews", "rating_pct": "Positive %", "subgenres": "Sub-genres",
        "developer": "Developer", "is_indie": "Indie", "genres_core": "Genres",
        "playtime_hours": "Avg playtime (h)", "is_hit": "Hit", "quadrant": "Group",
        "label": "Group", "games": "Games", "hits": "Hits", "hit_rate": "Hit rate %",
        "ci_low": "CI low %", "ci_high": "CI high %", "signal": "vs average"}


def dataframe(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def table(data, **kwargs):
    data = data.rename(columns=NICE)
    if "Sub-genres" in data:
        data = data.assign(**{"Sub-genres": data["Sub-genres"].str.replace("|", ", ", regex=False)})
    if "Genres" in data:
        data = data.assign(Genres=data["Genres"].str.replace("|", ", ", regex=False))
    dataframe(data, hide_index=True, **kwargs)


def card(name):
    """A pixel card section (styled through its st-key-card_* CSS class)."""
    return st.container(key=f"card_{name}")


def key_insights(name, lines):
    """Yellow game-dialog box at the bottom of a page or tab (2–4 bullets about it only)."""
    if not lines:
        return
    with st.container(key=f"msgbox_{name}"):
        st.markdown("<div class='msg-label'>KEY INSIGHTS</div>", unsafe_allow_html=True)
        st.markdown("\n".join(f"- {line}" for line in lines))


def page_header(title, tagline):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p class='tagline'>{tagline}</p></div>",
                unsafe_allow_html=True)


def nav_bar(current):
    """HOME / STORY MODE / HIT LAB / ARCADE on every page; the current one is pressed in."""
    with st.container(key="topnav"):
        for col, (path, label) in zip(st.columns(len(PAGES)), PAGES):
            slot = "navcur" if label == current else "nav"
            with col, st.container(key=f"{slot}_{Path(path).stem}"):
                st.page_link(path, label=label)


def filters_note():
    with st.sidebar:
        st.caption("The filters live in Story Mode. They do not affect this page.")


def pixel_icon(rows, colors, grid=16, size=64):
    """Pixel art as one inline SVG on an integer grid (no image files).

    Same-colour neighbours in a row are merged into one <rect>, and the SVG is shown at an
    exact multiple of the grid (64 px = 4 px per pixel) with crispEdges, so no seams appear
    between pixels at any zoom. The design is centred in a grid x grid square."""
    left, top = (grid - len(rows[0])) // 2, (grid - len(rows)) // 2
    rects = []
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            c, run = row[x], 1
            while x + run < len(row) and row[x + run] == c:
                run += 1
            if c != ".":
                rects.append(f"<rect x='{left + x}' y='{top + y}' width='{run}' height='1' "
                             f"fill='{colors[c]}'/>")
            x += run
    return (f"<svg class='picon' viewBox='0 0 {grid} {grid}' width='{size}' height='{size}' "
            f"shape-rendering='crispEdges' aria-hidden='true'>{''.join(rects)}</svg>")


# ------------------------------------------------------------------ data + model (cached, shared)
@st.cache_data
def load():
    return load_clean()


def data():
    """(full clean data 2020–2024, study cohort 2020–2023)."""
    if not C.CLEAN_FILE.exists():
        st.error("Cleaned data not found. Run `python -m src.cleaning` from the project root first.")
        st.stop()
    full = load()
    return full, A.study(full)


@st.cache_data
def load_pipeline():
    """Small files written by `python -m src.cleaning` (the raw file is never loaded here)."""
    d = C.PIPELINE_DIR
    if not (d / "raw_summary.json").exists():
        return None
    out = {name: pd.read_csv(d / f"{name}.csv") for name in
           ["raw_profile", "raw_sample", "clean_sample", "quality_examples",
            "missing_before_after"]}
    out["summary"] = json.loads((d / "raw_summary.json").read_text())
    log = pd.read_csv(C.CLEANING_LOG)
    # The content-filter step is shown with a neutral label only
    content = log["step"].str.contains("adult", case=False)
    log.loc[content, "step"] = "Content filter"
    log.loc[content, "reason"] = ""
    log.loc[log["step"] == "Raw file loaded", "step"] = "Raw file"
    out["log"] = log
    return out


@st.cache_resource
def load_hit_model():
    """Trained by `python -m src.model`; loaded once and shared by every page and session."""
    if not (C.MODEL_FILE.exists() and C.METRICS_FILE.exists()):
        return None, None
    return M.load_model()


@st.cache_data
def headline_numbers():
    """Three numbers for the title screen (study cohort 2020–2023)."""
    _, s = data()
    rate = s.groupby("is_indie")["is_hit"].mean() * 100
    return {"indie": rate[True], "other": rate[False],
            "few_reviews": s["few_reviews"].mean() * 100}
