"""STORY MODE: the analysis, as tabs. The sidebar filters apply to this page only."""
import pandas as pd
import streamlit as st

import charts
import common as U
from common import A, C, card, key_insights, show, table

df_full, df_study = U.data()
all_genres = sorted({g for s in df_study["genres_core"] for g in s.split("|") if g})
all_subgenres = sorted({g for s in df_study["subgenres"] for g in s.split("|") if g})

FEATURES = pd.DataFrame([
    ("review_count, few_reviews", "num_reviews_total", "-1 → 0 and flagged",
     "Visibility, without dropping unseen games"),
    ("rating_pct", "pct_pos_total", "Kept only with 10+ reviews", "Player satisfaction"),
    ("is_indie", "genres", "\"Indie\" in the genre list", "The developer's own label"),
    ("genres_core", "genres", "Keep the 9 core gameplay genres (multi-label)", "Genre comparison"),
    ("subgenres, has_tags", "tags (player votes)", "Top-10 tags mapped to 40 sub-genres",
     "Competition analysis"),
    ("price_band, is_free", "price", "Bins: Free, under $5, $5–10, $10–20, $20+",
     "Pricing comparison"),
    ("n_languages, arabic_support", "supported_languages", "Parsed list → count, Arabic flag",
     "Reach"),
    ("n_platforms", "windows, mac, linux", "Number of supported platforms", "Reach"),
    ("is_multiplayer", "categories", "Any multiplayer / co-op category", "Game mode"),
    ("has_achievements", "achievements", "More than 0", "Product feature"),
    ("owners_low, owners_high", "estimated_owners", "\"20000 - 50000\" split into two numbers",
     "A range, not sales"),
    ("release_year, release_quarter, days_on_market", "release_date", "From the date",
     "Control for game age"),
    ("developer, self_published", "developers, publishers",
     "First developer; developer = publisher", "Studio type"),
    ("visibility_pctile", "review_count, release_year", "Percentile within the release year",
     "Fair comparison across years"),
    ("quadrant", "review_count, rating_pct, is_hit", "Hit × 80%+ positive (50+ reviews)",
     "Loved vs Seen"),
], columns=["New feature", "Built from", "How", "Why"])

# ------------------------------------------------------------------ sidebar filters
FILTER_DEFAULTS = {"flt_years": (2020, 2023), "flt_group": "All games", "flt_price": 100,
                   "flt_genres": [], "flt_minrev": 0}
for k, v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(k, v)
with st.sidebar:
    st.divider()
    st.markdown("### Filters")
    years = st.slider("Release year", 2020, 2023, key="flt_years")
    group = st.selectbox("Studio type", ["All games", "Indie only", "Non-Indie only"],
                         key="flt_group")
    price_max = st.slider("Max price (USD)", 0, 100, step=5, key="flt_price",
                          help="100 = no limit")
    genres = st.multiselect("Steam genre", all_genres, placeholder="All genres", key="flt_genres")
    min_reviews = st.slider("Minimum reviews", 0, 500, step=10, key="flt_minrev",
                            help="Hide games with very few reviews")

f = df_study[df_study["release_year"].between(*years)
             & ((df_study["price"] <= price_max) | (price_max == 100))
             & (df_study["review_count"] >= min_reviews)]
if group == "Indie only":
    f = f[f["is_indie"]]
elif group == "Non-Indie only":
    f = f[~f["is_indie"]]
if genres:
    f = f[f["genres_core"].apply(lambda s: bool(set(s.split("|")) & set(genres)))]

with st.sidebar:
    st.caption(f"Showing **{len(f):,}** of {len(df_study):,} study games (2020–2023)")

U.page_header("STORY MODE", "What 36,000 Steam games reveal about Indie success on Steam")


def overview():
    c1, c2 = st.columns([3, 2])
    with c1, card("releases"):
        show(charts.releases_per_year(f))
    with c2, card("price"):
        show(charts.price_box(f))
        st.markdown("<p class='caveat'>Prices are the store price at the March 2025 snapshot, "
                    "not necessarily the launch price.</p>", unsafe_allow_html=True)
    with card("rating"):
        show(charts.rating_histogram(f))
    with card("summary"):
        st.markdown("#### Summary statistics")
        stats = f[["price", "review_count", "rating_pct", "playtime_hours", "n_languages",
                   "achievements"]].describe().T.round(1)
        stats.index = ["Price ($)", "Reviews", "Positive %", "Avg playtime (h)", "Languages",
                       "Achievements"]
        U.dataframe(stats)
        st.markdown("<p class='caveat'>Reviews are heavily skewed (a few huge games), so medians "
                    "(50%) describe a typical game better than means.</p>", unsafe_allow_html=True)

    rated = f[f["review_count"] >= C.MIN_REVIEWS_RATING]
    lines = [f"**{f['few_reviews'].mean() * 100:.0f}%** of these games have fewer than 10 "
             "reviews — visibility is the first problem most games face.",
             f"Overall hit rate in this selection: **{f['is_hit'].mean() * 100:.1f}%** "
             "(top 5% of reviews in each release year)."]
    if len(rated) > 30:
        rho = A.spearman(rated["price"], rated["rating_pct"])
        lines.append(f"Price and player rating are almost unrelated (Spearman ρ = {rho:.2f}): "
                     "paying more does not buy happier players.")
    band = A.rate_table(f, "price_band", min_n=30)
    if len(band):
        best = band.sort_values("hit_rate", ascending=False).iloc[0]
        lines.append(f"Highest hit rate by price: **{best['price_band']}** "
                     f"({best['hit_rate']:.1f}% of {int(best['games']):,} games).")
    key_insights("overview", lines)


def pipeline():
    P = U.load_pipeline()
    if P is None:
        st.info("Pipeline files not found. Run `python -m src.cleaning` from the project root.")
        return
    S, Q, log = P["summary"], P["quality_examples"], P["log"]

    def q(finding, label):
        return Q[(Q["finding"] == finding) & (Q["label"] == label)]["value"].iloc[0]

    def removed(step):
        return int(log.loc[log["step"] == step, "removed"].iloc[0])

    # 1) Before vs after
    c1, c2 = st.columns(2)
    with c1, card("raw"):
        st.markdown("#### Original file (Kaggle)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{S['raw_rows']:,}")
        m2.metric("Columns", f"{S['raw_columns']}")
        m3.metric("File size", f"{S['raw_size_mb']:.0f} MB")
        st.markdown("<p class='caveat'>All Steam apps in the March 2025 snapshot, "
                    "released 1997–2025. Never modified.</p>", unsafe_allow_html=True)
        with st.expander(f"All {S['raw_columns']} raw columns"):
            U.dataframe(P["raw_profile"].rename(columns={
                "column": "Column", "dtype": "Type", "missing_pct": "Missing %",
                "unique_values": "Unique values", "example": "Example"}), hide_index=True)
    with c2, card("clean"):
        st.markdown("#### Clean file")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{S['clean_rows']:,}")
        m2.metric("Columns", f"{S['clean_columns']}")
        m3.metric("File size", f"{S['clean_size_mb']:.0f} MB")
        st.markdown("<p class='caveat'>Games released 2020–2024 (2020–2023 study + 2024 "
                    "check), with parsed lists and new features.</p>", unsafe_allow_html=True)

    # 2) Cleaning funnel
    with card("funnel"):
        show(charts.cleaning_funnel(log))

    # 3) Raw vs clean preview
    with card("preview"):
        st.markdown("#### Raw vs clean preview")
        st.markdown("**Original rows** (raw file, a few columns)")
        U.dataframe(P["raw_sample"], hide_index=True)
        st.markdown("**The same games after cleaning**")
        U.dataframe(P["clean_sample"], hide_index=True)
        st.markdown(
            "- **Lists parsed:** text like `['Action', 'Indie']` becomes real lists → "
            "`is_indie`, `genres_core`, `n_languages`.\n"
            "- **-1 handled:** `num_reviews_total = -1` becomes `review_count = 0` with "
            "`few_reviews = True` (see the last row).\n"
            "- **Ranges split:** `estimated_owners` \"5000000 - 10000000\" → "
            "`owners_low`, `owners_high`.\n"
            "- **Tags mapped:** the top-10 player tags become `subgenres` "
            "(e.g. Action Roguelike → Roguelike).")

    # 4) Data quality findings
    st.markdown("")
    tests = Q[Q["finding"] == "playtests"]
    dups = Q[Q["finding"] == "duplicates"]
    r1c1, r1c2 = st.columns(2)
    with r1c1, card("find_sentinel"):
        st.markdown("#### -1 means fewer than 10 reviews")
        st.markdown(f"<div class='big'>{int(q('sentinel', 'Rows with -1')):,} rows "
                    f"({q('sentinel', 'Share of rows with -1 (%)'):.1f}%)</div>",
                    unsafe_allow_html=True)
        st.markdown(f"The smallest real review count is "
                    f"**{int(q('sentinel', 'Smallest real review count'))}**: Steam shows no "
                    "review summary below 10, so `-1` is not missing data. These games are "
                    "kept as 0 reviews and flagged `few_reviews`.")
    with r1c2, card("find_sources"):
        st.markdown("#### Two review sources disagree")
        st.markdown(f"<div class='big'>{q('sources', 'Identical counts (%)'):.1f}% "
                    "identical</div>", unsafe_allow_html=True)
        st.markdown(f"SteamSpy's `positive + negative` matches Steam's `num_reviews_total` "
                    f"in only {q('sources', 'Identical counts (%)'):.1f}% of "
                    f"{int(q('sources', 'Games with both counts')):,} games, and SteamSpy "
                    f"covers {q('sources', 'SteamSpy coverage 2020 (%)'):.0f}% of 2020 games "
                    f"but {q('sources', 'SteamSpy coverage 2023 (%)'):.0f}% of 2023 games. "
                    "Only Steam's own counts are used.")
    r2c1, r2c2 = st.columns(2)
    with r2c1, card("find_tests"):
        st.markdown("#### Playtests copy main-game reviews")
        st.markdown(f"<div class='big'>{removed('Remove test builds'):,} removed</div>",
                    unsafe_allow_html=True)
        st.markdown("Test builds show the main game's review count, so one success would "
                    "be counted twice:\n" + "\n".join(
                        f"- {r.label}: {int(r.value):,} reviews" for r in tests.itertuples()))
    with r2c2, card("find_dups"):
        st.markdown("#### Duplicate editions")
        st.markdown(f"<div class='big'>{removed('Remove duplicate editions'):,} removed</div>",
                    unsafe_allow_html=True)
        st.markdown("Extra store pages (editions, bundles) share one review pool with the "
                    "main game. Only the first page is kept:\n" + "\n".join(
                        f"- {r.label}: {int(r.detail)} extra page"
                        f"{'s' if int(r.detail) > 1 else ''}, {int(r.value):,} reviews"
                        for r in dups.itertuples()))

    # 5) Feature engineering
    with card("features"):
        st.markdown("#### Feature engineering")
        rows_h = (len(FEATURES) + 1) * 35 + 3     # show every row, no inner scrolling
        U.dataframe(FEATURES, hide_index=True, height=rows_h)
        st.markdown(
            "**Target `is_hit`:** a game in the **top 5% of review count among games "
            "released the same year** (games with fewer than 10 reviews are never hits). "
            "Comparing within a year is fair to newer games.\n\n"
            "**Robustness checks:** top 10% (`is_hit_top10`), top 5% within the release "
            "quarter (`is_hit_by_quarter`), and the 2024 cohort. The Indie gap holds under "
            "all of them.")

    # 6) Missing values
    with card("missing"):
        show(charts.missing_compare(P["missing_before_after"]))
        st.markdown("<p class='caveat'>Review count: <code>-1</code> is handled, so nothing is "
                    "missing. Rating % is left empty on purpose below 10 reviews (too few to "
                    "trust). Player tags cannot be invented, so games without tags stay "
                    "without sub-genres.</p>", unsafe_allow_html=True)

    steps = log[~log["step"].isin(["Raw file", "Content filter"])]
    big = steps.loc[steps["removed"].idxmax()]
    n_new = sum(len(x.split(", ")) for x in FEATURES["New feature"])
    key_insights("pipeline", [
        f"**{S['raw_rows']:,}** raw rows become **{S['clean_rows']:,}** clean rows "
        f"({100 * S['clean_rows'] / S['raw_rows']:.0f}% kept); {len(df_study):,} of them form "
        "the 2020–2023 study cohort.",
        f"Biggest cleaning step: **{big['step']}** ({int(big['removed']):,} rows removed).",
        f"**{q('sentinel', 'Share of rows with -1 (%)'):.1f}%** of raw rows have "
        "`num_reviews_total = -1` — fewer than 10 reviews, kept as 0 instead of dropped.",
        f"Cleaning adds **{n_new} new features**, from simple flags to the `is_hit` target."])


def indie_reality():
    with card("gap"):
        show(charts.indie_gap(df_full))
        st.markdown("<p class='caveat'>'Indie' = the Steam genre label chosen by the developer. "
                    "It does not prove studio size or budget.</p>", unsafe_allow_html=True)
    share = A.indie_share_of_hits(df_full)
    c1, c2 = st.columns(2)
    c1.metric("Indie share of all games", f"{share['share_of_games']:.0f}%")
    c2.metric("Indie share of hits", f"{share['share_of_hits']:.0f}%",
              f"{share['share_of_hits'] - share['share_of_games']:.0f} pts", delta_color="normal")
    gap = A.indie_gap(df_full)
    rates = gap.pivot(index="definition", columns="group", values="hit_rate")
    main = rates.loc[rates.index.str.contains("main")].iloc[0]
    holds = int((rates["Indie"] < rates["Non-Indie"]).sum())
    key_insights("indie", [
        f"Indie games become hits **{main['Indie']:.1f}%** of the time vs "
        f"**{main['Non-Indie']:.1f}%** for non-Indie games.",
        f"Indie games are **{share['share_of_games']:.0f}%** of releases but only "
        f"**{share['share_of_hits']:.0f}%** of hits.",
        (f"The gap holds under **all {len(rates)}** hit definitions"
         if holds == len(rates) else f"The gap holds under {holds} of {len(rates)} definitions")
        + " (top 5%, top 10%, per quarter, 2024 cohort) — not an artefact of one definition."])


def opportunity():
    with card("opportunity"):
        c1, c2, c3 = st.columns([2, 2, 2])
        level = c1.radio("Group games by", ["Sub-genre (player tags)", "Steam genre"],
                         horizontal=True)
        scope = c2.radio("Games", ["Indie only", "All games"], horizontal=True)
        min_n = c3.slider("Minimum games per group", 30, 300, 80, step=10)
        col = "subgenres" if level.startswith("Sub") else "genres_core"
        src = f[f["is_indie"]] if scope == "Indie only" else f
        t, base = A.opportunity_table(src, col=col, indie_only=False, min_n=min_n)
        if t.empty:
            st.info("Not enough games per group with these filters. Lower the minimum or widen "
                    "filters.")
        else:
            show(charts.opportunity_map(t, base, level.split(" (")[0].lower()))
            st.markdown("<p class='caveat'>Green/orange only when the 95% CI is entirely "
                        "above/below the average.</p>", unsafe_allow_html=True)
            with st.expander("See the table"):
                table(t.round(1)[["label", "games", "hits", "hit_rate", "ci_low", "ci_high",
                                  "signal"]])
    if t.empty:
        return

    def fmt(rows):
        return ", ".join(f"{r.label} ({r.hit_rate:.1f}%)" for r in rows.itertuples())
    top = t[t["signal"] == "Above average"].sort_values("hit_rate", ascending=False).head(3)
    low = t[t["signal"] == "Below average"].sort_values("hit_rate").head(3)
    lines = []
    if len(top):
        lines.append(f"Clearly above the {base:.1f}% average: **{fmt(top)}**.")
    if len(low):
        lines.append(f"Clearly below: **{fmt(low)}**.")
    lines.append("Few games + a high hit rate is a signal worth investigating, not a "
                 "guaranteed gap — a small niche can also mean small demand.")
    key_insights("opportunity", lines)


def loved_vs_seen():
    with card("loved"):
        show(charts.popularity_vs_satisfaction(f))
    q = f[f["quadrant"] != "Not enough reviews"]
    if not len(q):
        return
    counts = q["quadrant"].value_counts()
    for c, name in zip(st.columns(4), C.QUADRANT_COLORS):
        c.metric(name, f"{counts.get(name, 0):,}")
    gems = counts.get("Hidden Gem", 0)
    loved = counts.get("Popular & Loved", 0)
    with card("gems"):
        st.markdown("#### Top Hidden Gems in this selection")
        gems_df = q[q["quadrant"] == "Hidden Gem"].sort_values(
            ["rating_pct", "review_count"], ascending=False).head(15)
        table(gems_df[["name", "release_year", "price", "review_count", "rating_pct",
                       "subgenres"]])
    lines = []
    if loved:
        lines.append(f"For every **Popular & Loved** game there are about **{gems / loved:.1f} "
                     "Hidden Gems**: well-rated games that never reached a large audience. "
                     "Quality alone does not guarantee visibility.")
    divisive = q[q["quadrant"] == "Popular but Divisive"]
    if len(divisive):
        lines.append(f"**{(~divisive['is_indie']).mean() * 100:.0f}%** of the "
                     f"{len(divisive):,} *Popular but Divisive* games are non-Indie: big "
                     "games can succeed without being loved.")
    key_insights("loved", lines)


def hit_patterns():
    with card("history"):
        st.markdown("#### Historical comparison")
        st.markdown("Describe your game and see how **similar Indie games released 2020–2023** "
                    "performed. This is a historical comparison, **not a prediction** (the "
                    "prediction lives in HIT LAB).")
        c1, c2, c3, c4 = st.columns(4)
        lab_genre = c1.selectbox("Steam genre", ["Any"] + all_genres)
        lab_sub = c2.selectbox("Sub-genre", ["Any"] + all_subgenres)
        lab_price = c3.selectbox("Price band", ["Any"] + C.PRICE_LABELS)
        lab_mode = c4.selectbox("Mode", ["Any", "Has multiplayer", "Single-player only"])
        similar, res = A.similar_games(
            df_full, genre=None if lab_genre == "Any" else lab_genre,
            subgenre=None if lab_sub == "Any" else lab_sub,
            price_band=None if lab_price == "Any" else lab_price,
            multiplayer=None if lab_mode == "Any" else lab_mode == "Has multiplayer")
        base_rate = df_study.query("is_indie")["is_hit"].mean() * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Similar Indie games", f"{res['games']:,}")
        if res["games"] == 0:
            st.warning("No similar games found. Try a broader combination.")
        else:
            m2.metric("Became hits", f"{res['hits']:,}")
            m3.metric("Hit rate", f"{res['hit_rate']:.1f}%",
                      f"{res['hit_rate'] - base_rate:+.1f} pts vs Indie average")
            st.progress(min(res["hit_rate"] / 100, 1.0))
            st.caption(f"95% confidence interval: {res['ci_low']:.1f}% – {res['ci_high']:.1f}% · "
                       f"Indie average {base_rate:.1f}%")
            if not res["enough_data"]:
                st.warning(f"Insufficient data: only {res['games']} similar games "
                           f"(fewer than {C.MIN_SAMPLE}). The rate above is not reliable.")
    if res["games"] and res["hits"]:
        with card("hits"):
            st.markdown("#### Hits that match this profile")
            hits = similar[similar["is_hit"]].sort_values("review_count", ascending=False)
            table(hits[["name", "release_year", "price", "review_count", "rating_pct",
                        "subgenres"]].head(10))

    with card("common"):
        st.markdown("#### What do Indie hits have in common?")
        effects = A.feature_effects(df_full)
        factor = st.radio("Factor", effects["factor"].unique().tolist(), horizontal=True)
        show(charts.feature_effect(effects, factor))
        if factor == "Price":
            st.markdown("<p class='caveat'>Price is a confounder: it stands for budget and scope, "
                        "not a cause of success.</p>", unsafe_allow_html=True)

    spread = effects.groupby("factor")["hit_rate"].agg(["max", "min"])
    strongest = (spread["max"] - spread["min"]).idxmax()
    g = effects[effects["factor"] == strongest].set_index("level")["hit_rate"]
    lines = []
    if res["games"]:
        lines.append(f"Your profile: **{res['games']:,}** similar Indie games, "
                     f"**{res['hit_rate']:.1f}%** became hits (Indie average {base_rate:.1f}%)"
                     + ("." if res["enough_data"] else " — too few games to trust."))
    lines.append(f"Strongest factor among Indie hits: **{strongest}** — {g.idxmax()} "
                 f"{g.max():.1f}% vs {g.idxmin()} {g.min():.1f}% hit rate.")
    lines.append("These are associations, not causes: successful games often add languages, "
                 "achievements and platforms *after* launch, and higher prices usually mean "
                 "bigger budgets.")
    key_insights("patterns", lines)


def explorer():
    with card("explorer"):
        st.markdown("#### Game Explorer")
        query = st.text_input("Search a game by name",
                              placeholder="e.g. Stardew, Vampire, Hades…")
        e = f[f["name"].str.contains(query, case=False, na=False, regex=False)] if query else f
        sort_by = st.selectbox("Sort by", ["review_count", "rating_pct", "price", "release_date"])
        e = e.sort_values(sort_by, ascending=False)
        st.caption(f"{len(e):,} games")
        table(e[["name", "developer", "release_date", "is_indie", "price", "genres_core",
                 "subgenres", "review_count", "rating_pct", "playtime_hours", "is_hit",
                 "quadrant"]].head(500))
        st.download_button("Download selection (CSV)", e.to_csv(index=False).encode(),
                           "steam_selection.csv", "text/csv")
    if len(e):
        what = f"match “{query}”" if query else "match the sidebar filters"
        key_insights("explorer", [
            f"**{len(e):,}** games {what}; **{int(e['is_hit'].sum()):,}** of them are hits "
            f"({e['is_hit'].mean() * 100:.1f}%)."])


if f.empty:
    st.warning("No games match these filters. Try widening them in the sidebar.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Games in selection", f"{len(f):,}")
    k2.metric("Indie share", f"{f['is_indie'].mean() * 100:.0f}%")
    k3.metric("Hit rate", f"{f['is_hit'].mean() * 100:.1f}%",
              help="Share of games in the top 5% of reviews for their release year")
    k4.metric("Fewer than 10 reviews", f"{f['few_reviews'].mean() * 100:.0f}%")

    TABS = {"Overview": overview, "Data Pipeline": pipeline, "Indie Reality": indie_reality,
            "Opportunity Map": opportunity, "Loved vs Seen": loved_vs_seen,
            "Hit Patterns": hit_patterns, "Game Explorer": explorer}
    for tab, render in zip(st.tabs(list(TABS)), TABS.values()):
        with tab:
            render()
