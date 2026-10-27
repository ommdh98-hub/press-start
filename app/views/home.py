"""HOME: the title screen with the three game modes."""
import streamlit as st

import common as U

K = U.K
ICONS = {
    # open book
    "story": (["..pppp..pppp..", ".pffffppffffp.", "pfmmffppfmmffp", "pffffmppffmffp",
               "pfmmffppfmmffp", "pffmffppffffmp", "pfmmffppfmmffp", "pffffppppffffp",
               "pppppp..pppppp", "......pp......"],
              {"p": K["indie"], "f": K["text"], "m": K["muted"]}),
    # flask with a bubbling potion
    "lab": (["...mmmmmm...", "....f..f....", "....f..f....", "....f..f....", "...f....f...",
             "..f......f..", "..fyyyyyyf..", ".fyyyyyyyyf.", ".fyyfyyyyyf.", ".fyyyyyfyyf.",
             ".fyyyyyyyyf.", "..ffffffff.."],
            {"f": K["text"], "y": K["accent"], "m": K["muted"]}),
    # game pad
    "arcade": (["..tttttttttt..", ".tttttttttttt.", "tttftttttttytt", "ttffftttttttot",
                "tttftttttttott", "tttttttttttttt", "ttttt....ttttt", ".ttt......ttt."],
               {"t": K["other"], "f": K["text"], "y": K["accent"], "o": K["warn"]}),
}
MODES = [
    ("story", "views/story.py", "STORY MODE", "Explore 36,000 games",
     "The analysis: Indie hit rates, opportunities, hidden gems and the data pipeline.",
     "START STORY"),
    ("lab", "views/hit_lab.py", "HIT LAB", "Predict your next hit",
     "Build a game idea and get its hit chance, rank and power-ups.",
     "OPEN HIT LAB"),
    ("arcade", "views/arcade.py", "ARCADE", "Play Indie Run",
     "A pixel endless runner whose levels follow real Indie release counts.",
     "PLAY"),
]

st.markdown("<div class='hero title'><h1>PRESS START</h1><p class='tagline'>What 36,000 Steam "
            "games reveal about Indie success</p></div>", unsafe_allow_html=True)

for col, (key, path, title, tag, desc, button) in zip(st.columns(3), MODES):
    with col, st.container(key=f"mode_{key}"):
        rows, colors = ICONS[key]
        st.markdown(U.pixel_icon(rows, colors)
                    + f"<div class='mode-head'><div class='mode-title'>{title}</div>"
                    f"<div class='mode-tag'>{tag}</div></div><div class='mode-desc'>{desc}</div>",
                    unsafe_allow_html=True)
        st.page_link(path, label=button)

st.markdown("")
h = U.headline_numbers()
_, metrics = U.load_hit_model()
n1, n2, n3 = st.columns(3)
n1.metric("Hit rate: Indie vs non-Indie", f"{h['indie']:.1f}% vs {h['other']:.1f}%",
          help="Hit = top 5% of review count among games released the same year (2020–2023)")
n2.metric("Games with fewer than 10 reviews", f"{h['few_reviews']:.0f}%")
n3.metric("Hit Predictor test ROC-AUC",
          f"{metrics['test']['roc_auc']:.2f}" if metrics else "not trained",
          help="On unseen test games; 0.5 = coin flip, 1.0 = perfect")
