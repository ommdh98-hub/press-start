"""HIT LAB: the Hit Predictor as a mini-game, plus a small model card."""
import numpy as np
import streamlit as st

import common as U
from common import C, K, M, card, key_insights, table

df_full, df_study = U.data()
bundle, metrics = U.load_hit_model()
U.filters_note()

RANK_COLORS = {"S": K["accent"], "A": K["other"], "B": K["single"], "C": K["neutral"],
               "D": K["warn"]}
IDEA_DEFAULTS = {"sub": [], "gen": [], "price": "$10–20", "plat": 1, "mp": "No", "ea": "No",
                 "month": "Oct"}
FEATURE_LABELS = {"n_platforms": "Platforms (1–3)", "is_indie": "Indie label",
                  "is_early_access": "Early Access", "is_multiplayer": "Multiplayer",
                  "price_band": "Price band (5 bands)", "release_month": "Release month",
                  "genres_core": "Steam genres", "subgenres": "Sub-genres (player tags)"}


@st.cache_data
def indie_score_scale(_bundle, model_mtime):
    """Predicted chance of every Indie game in the study cohort (computed once per model)."""
    return M.indie_scores(_bundle, df_full)


def roll_dice(prefix, sub_opts, gen_opts):
    """Button callback: fill one idea panel with the profile of a random real Indie game."""
    idea = M.random_idea(df_full, np.random.default_rng(), sub_opts, gen_opts)
    ss = st.session_state
    ss[f"{prefix}_sub"], ss[f"{prefix}_gen"] = idea["subgenres"], idea["genres"]
    ss[f"{prefix}_price"], ss[f"{prefix}_plat"] = idea["price_band"], idea["n_platforms"]
    ss[f"{prefix}_mp"] = "Yes" if idea["multiplayer"] else "No"
    ss[f"{prefix}_ea"] = "Yes" if idea["early_access"] else "No"
    ss[f"{prefix}_month"] = M.MONTHS[idea["month"] - 1]


def idea_panel(prefix, sub_opts, gen_opts):
    """Inputs for one game idea; values live in st.session_state under '<prefix>_*'."""
    for k, v in IDEA_DEFAULTS.items():
        st.session_state.setdefault(f"{prefix}_{k}", v)
    sub = st.multiselect("Sub-genres (max 3)", sub_opts, max_selections=3,
                         key=f"{prefix}_sub", placeholder="Choose up to 3")
    gen = st.multiselect("Steam genres", gen_opts, key=f"{prefix}_gen",
                         placeholder="Choose genres")
    b1, b2 = st.columns(2)
    price = b1.selectbox("Price band", C.PRICE_LABELS, key=f"{prefix}_price")
    plat = b2.slider("Platforms", 1, 3, key=f"{prefix}_plat")
    b3, b4, b5 = st.columns(3)
    mp = b3.radio("Multiplayer", ["No", "Yes"], horizontal=True, key=f"{prefix}_mp")
    ea = b4.radio("Early Access", ["No", "Yes"], horizontal=True, key=f"{prefix}_ea")
    month = b5.selectbox("Release month", M.MONTHS, key=f"{prefix}_month")
    st.button("ROLL THE DICE", key=f"{prefix}_dice", on_click=roll_dice,
              args=(prefix, sub_opts, gen_opts))
    return {"price_band": price, "n_platforms": plat, "multiplayer": mp == "Yes",
            "early_access": ea == "Yes", "month": M.MONTHS.index(month) + 1,
            "genres": list(gen), "subgenres": list(sub)}


def score(idea, scale):
    p = M.predict_many(bundle, [idea])[0]
    pct = M.percentile(p, scale)
    return {"p": p * 100, "pct": pct, "rank": M.rank_of(pct)}


def rank_badge(res):
    color = RANK_COLORS[res["rank"]]
    return (f"<div class='rankrow'><div><div class='rank-label'>RANK</div>"
            f"<div class='rank' style='color:{color};border-color:{color}'>{res['rank']}</div>"
            f"</div><div><div class='better'>Better than {int(res['pct'])}% of Indie games</div>"
            f"<div class='chance-small'>Hit chance: {res['p']:.1f}% · "
            f"{res['p'] / metrics['indie_base_rate_pct']:.1f}× the Indie average "
            f"({metrics['indie_base_rate_pct']:.1f}%)</div></div></div>")


def idea_result(sub_opts, gen_opts, scale):
    """One game idea: inputs on the left, its rank, power-ups and score drivers on the right."""
    c1, c2 = st.columns([5, 6])
    with c1, card("idea"):
        st.markdown("#### Your idea")
        idea = idea_panel("pred", sub_opts, gen_opts)
    res = score(idea, scale)
    rank = res["rank"]
    moved = M.explain(bundle, M.make_row(**idea))
    up = moved[moved["effect"] > 0].head(3)["factor"].tolist()
    down = moved[moved["effect"] < 0].sort_values("effect").head(3)["factor"].tolist()
    with c2, card("result"):
        st.markdown("#### Result")
        st.markdown(rank_badge(res), unsafe_allow_html=True)
        st.progress(min(res["pct"] / 100, 1.0))
        st.caption("Power level: percentile among 2020–2023 Indie games")
        st.markdown("**What moved the score**")
        st.markdown("<div class='moved'>"
                    + "".join(f"<p style='color:{K['other']}'>+ {x}</p>" for x in up)
                    + "".join(f"<p style='color:{K['warn']}'>− {x}</p>" for x in down)
                    + "</div>", unsafe_allow_html=True)
        ups = M.power_ups(bundle, idea, sub_opts)
        st.markdown("**Power-ups**")
        if len(ups):
            html = ""
            for r in ups.itertuples():
                new_rank = M.rank_of(M.percentile(r.p, scale))
                step = f"{rank} to {new_rank}" if new_rank != rank else f"still {rank}"
                html += (f"<p class='pu'><span class='pu-tag'>POWER-UP</span> {r.change} "
                         f"-&gt; {step} (+{r.gain_pts:.1f} pts)</p>")
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("No single change raises this idea by 0.5 points or more.")
        st.markdown("<p class='caveat'>Power-ups show patterns in past games, not guaranteed "
                    "causes.</p>", unsafe_allow_html=True)

    pool = df_study[df_study["is_hit"] & df_study["is_indie"]]
    chosen = set(idea["subgenres"]) if idea["subgenres"] else set(idea["genres"])
    col = "subgenres" if idea["subgenres"] else "genres_core"
    if chosen:
        overlap = pool[col].apply(lambda x: len(chosen & set(x.split("|"))))
        similar_hits = pool.assign(_o=overlap)[overlap > 0].sort_values(
            ["_o", "review_count"], ascending=False).head(5)
        if len(similar_hits):
            with card("similar"):
                st.markdown("#### Real Indie hits with similar "
                            f"{'sub-genres' if idea['subgenres'] else 'genres'}")
                table(similar_hits[["name", "release_year", "price", "review_count",
                                    "rating_pct", "subgenres"]])
    return res


def model_card():
    t = metrics["test"]
    with card("model"):
        st.markdown("#### Model card")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test ROC-AUC", f"{t['roc_auc']:.3f}",
                  help=f"{metrics['rows']['test']:,} unseen games, used once")
        m2.metric("CV ROC-AUC", f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}",
                  help="5-fold cross-validation on the training games")
        m3.metric("Train/test gap", f"{metrics['train_auc'] - t['roc_auc']:+.3f}",
                  help=f"Train {metrics['train_auc']:.3f} vs test {t['roc_auc']:.3f}: "
                       "no overfitting")
        f1, f2 = st.columns(2)
        f1.markdown("**Features used** (known before launch)\n"
                    + "\n".join(f"- {FEATURE_LABELS.get(x, x)}" for x in metrics["features"]))
        f2.markdown(
            "**Features excluded, and why**\n"
            "- Languages, achievements, DLC: often added *after* a game succeeds "
            "(reverse causality)\n"
            "- Reviews, rating, playtime, owners, peak players: measured after launch\n"
            "- Visibility percentile, quadrant, other hit labels: built from the review count, "
            "i.e. from the answer (leakage)")
        st.markdown(f"<p class='caveat'>L2 logistic regression with balanced classes and "
                    f"calibrated probabilities, trained on {metrics['rows']['train']:,} games "
                    f"(2020–2023, stratified 80/20 split). Time check: trained on "
                    f"{metrics['time_check']['train_years']}, it scores ROC-AUC "
                    f"{metrics['time_check']['roc_auc']:.3f} on "
                    f"{metrics['time_check']['test_year']}.</p>", unsafe_allow_html=True)


U.page_header("HIT LAB", "Build a game idea from choices you make before launch and see how it "
                         "ranks against every Indie game released 2020–2023")
if bundle is None:
    st.info("Model not found. Run `python -m src.model` from the project root.")
else:
    names = bundle["feature_names"]
    sub_opts = sorted(n.split(": ", 1)[1] for n in names if n.startswith("Sub-genre: "))
    gen_opts = sorted(n.split(": ", 1)[1] for n in names if n.startswith("Steam genre: "))
    scale = indie_score_scale(bundle, C.MODEL_FILE.stat().st_mtime)
    res = idea_result(sub_opts, gen_opts, scale)
    st.markdown(f"<p class='caveat'>Estimate based on patterns in 2020–2023 Steam games. "
                f"Not a guarantee. Test ROC-AUC = {metrics['test']['roc_auc']:.2f} on "
                "unseen games. Price reflects budget and scope (bigger teams, more "
                "content, more marketing). Raising the price of the same game will not "
                "make it a hit.</p>", unsafe_allow_html=True)
    model_card()
    key_insights("hitlab", [
        f"Your idea: rank **{res['rank']}**, better than **{int(res['pct'])}%** of Indie games, "
        f"**{res['p']:.1f}%** hit chance (**{res['p'] / metrics['indie_base_rate_pct']:.1f}×** "
        f"the Indie average of {metrics['indie_base_rate_pct']:.1f}%).",
        f"The model ranks unseen games well (test ROC-AUC **{metrics['test']['roc_auc']:.2f}**, "
        f"**{metrics['time_check']['roc_auc']:.2f}** on the next year), but chances stay low: "
        f"its top 10% of games became hits **{metrics['test']['top10_hit_rate_pct']:.1f}%** of "
        f"the time vs {metrics['test']['base_rate_pct']:.1f}% overall.",
        "Power-ups are patterns in past games, not causes; price stands for budget, so it is "
        "never suggested as a power-up."])
