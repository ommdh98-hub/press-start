"""ARCADE: Indie Run, a canvas game fed with real numbers from the clean data."""
import base64
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import common as U
from common import A, C, card, key_insights

GAME_FILE = Path(__file__).resolve().parents[1] / "game" / "indie_run.html"
LOGO_FILE = GAME_FILE.parent / "assets" / "tuwaiq_logo.png"   # official logo, used as is
# The Flag of Saudi Arabia, the public-domain file from Wikimedia Commons, used as is
FLAG_FILE = GAME_FILE.parent / "assets" / "saudi_flag.png"
GAME_MAX_W = 1000                                  # same cap as MAX_W in indie_run.html
GAME_FRAME = 10                                    # canvas border (2 x 3 px) + shadow (4 px)
GAME_FRAME_H = GAME_MAX_W * 9 // 16 + GAME_FRAME   # 572 px


@st.cache_data
def arcade_numbers():
    """Real numbers for the Indie Run game (full clean data; sidebar filters do not apply)."""
    data, s = U.data()
    releases = data[data["is_indie"]].groupby("release_year").size()
    subs, _ = A.opportunity_table(data, col="subgenres", indie_only=True, min_n=80)
    top = subs.sort_values("hit_rate", ascending=False).head(3)
    cozy = subs.loc[subs["label"] == "Cozy", "hit_rate"]
    effects = A.feature_effects(data).set_index(["factor", "level"])["hit_rate"]
    return {
        "releases": {str(int(y)): int(n) for y, n in releases.items()},
        "indie_hit_rate": round(float(s.loc[s["is_indie"], "is_hit"].mean() * 100), 1),
        "overall_hit_rate": round(float(s["is_hit"].mean() * 100), 1),
        "few_reviews_pct": round(float(s["few_reviews"].mean() * 100), 1),
        "top_subgenres": [{"name": r.label, "hit_rate": round(float(r.hit_rate), 1),
                           "games": int(r.games)} for r in top.itertuples()],
        "powerups": {
            "multiplayer": round(float(effects[("Mode", "Has multiplayer")]), 1),
            "single_player": round(float(effects[("Mode", "Single-player only")]), 1),
            "languages_10plus": round(float(effects[("Languages", "10+")]), 1),
            "one_language": round(float(effects[("Languages", "1")]), 1),
            "cozy": round(float(cozy.iloc[0]), 1) if len(cozy) else None,
        },
    }


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def game_html(numbers):
    """The game page with the real numbers in GAME_DATA and the logo and flag as data URIs
    (the component iframe cannot load files from the repo)."""
    payload = json.dumps(numbers).replace("</", "<\\/")
    html = re.sub(r"/\*GAME_DATA\*/.*?/\*END_GAME_DATA\*/",
                  lambda m: f"/*GAME_DATA*/{payload}/*END_GAME_DATA*/",
                  GAME_FILE.read_text(encoding="utf-8"), count=1, flags=re.S)
    for marker, path in (("TUWAIQ_LOGO", LOGO_FILE), ("SAUDI_FLAG", FLAG_FILE)):
        uri = data_uri(path)
        html = re.sub(rf"/\*{marker}\*/.*?/\*END_{marker}\*/",
                      lambda m: f'/*{marker}*/"{uri}"/*END_{marker}*/', html, count=1, flags=re.S)
    return html


U.filters_note()
U.page_header("ARCADE", "INDIE RUN: survive five years of Steam releases")
game = arcade_numbers()
rel, pw, top3 = game["releases"], game["powerups"], game["top_subgenres"]
first, last = min(rel), max(rel)
with card("arcade"):
    # The game fits itself inside this frame (16:9, at most GAME_MAX_W wide); the frame is
    # sized for that maximum plus the canvas border and shadow, and centred in the card.
    components.html(game_html(game), height=GAME_FRAME_H)
    st.markdown("<p class='caveat game-note'>Click the game to focus it; keys only reach the "
                "game while it has focus. Uses the full 2020–2024 data; the sidebar filters do "
                "not apply here. Flag of Saudi Arabia: public-domain image from Wikimedia "
                "Commons.</p>", unsafe_allow_html=True)
with card("arcade_data"):
    st.markdown("#### How the game uses the data")
    st.markdown(
        f"- **Difficulty follows real release counts.** Each 15-second level is a release "
        f"year. Speed and obstacle density grow with that year's Indie releases in the clean "
        f"data: {rel[first]:,} in {first} → {rel[last]:,} in {last}. A final Tuwaiq Academy "
        f"level follows; reaching 10,000 points completes the run.\n"
        f"- **Power-ups come from factors associated with Indie hits.** Multiplayer "
        f"({pw['multiplayer']}% hit rate vs {pw['single_player']}% single-player), "
        f"Localization (10+ languages: {pw['languages_10plus']}% vs {pw['one_language']}% "
        f"with one) and Cozy ({pw['cozy']}%, the top Indie sub-genre). These are "
        f"associations, not causes; languages are often added after launch.\n"
        f"- **Game-over facts are real numbers** from the {C.STUDY_YEARS[0]}–"
        f"{C.STUDY_YEARS[1]} cohort, such as the Indie hit rate and the share of games with "
        f"fewer than 10 reviews.")
key_insights("arcade", [
    "Indie releases per year: " + ", ".join(f"{y}: **{n:,}**" for y, n in rel.items())
    + f" — the {last} level is the busiest.",
    f"Only **{game['indie_hit_rate']}%** of Indie games become hits, vs "
    f"**{game['overall_hit_rate']}%** of all games; **{game['few_reviews_pct']:.0f}%** of "
    f"games get fewer than 10 reviews.",
    "Top Indie sub-genres by hit rate (80+ games): " + ", ".join(
        f"**{t['name']}** ({t['hit_rate']}%, {t['games']:,} games)" for t in top3) + "."])
