"""Press Start — Indie games on Steam.

Run from the project root:
    streamlit run app/app.py

This file is the router: shared style, sidebar text and the top navigation bar.
The four pages live in app/views/ (HOME, STORY MODE, HIT LAB, ARCADE).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

import common as U  # noqa: E402

st.set_page_config(layout="wide")

pages = [st.Page(path, title=f"Press Start | {label.title()}", url_path=Path(path).stem,
                 default=(label == "HOME"))
         for path, label in U.PAGES]
page = st.navigation(pages, position="hidden")      # own menu below, built-in one hidden
current = dict(zip((p.title for p in pages), (label for _, label in U.PAGES)))[page.title]

# Keep filter and Hit Predictor values when the user visits another page and comes back
for k in list(st.session_state.keys()):
    if k.startswith(U.PERSIST_PREFIXES) and not k.endswith("_dice"):
        st.session_state[k] = st.session_state[k]

U.inject_css()
full, _ = U.data()
with st.sidebar:
    st.markdown("## Press Start")
    st.markdown(
        f"**Dataset:** [Steam Games Dataset]({U.C.DATASET_URL}) (Kaggle, March 2025 snapshot). "
        f"After cleaning: **{len(full):,} games** released 2020–2024 with price, genres, "
        "player tags, reviews, playtime and features.\n\n"
        "**Hit** = a game in the **top 5% of review count** among games released the same year. "
        "Reviews measure *visibility*, not sales.")
U.nav_bar(current)
page.run()

with st.sidebar:
    st.caption("Source: Steam Games Dataset (artermiloff, Kaggle, March 2025). Review counts "
               "measure visibility, not sales. Estimated owners are ranges and are not used as "
               "sales.")
