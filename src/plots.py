"""Static charts (Matplotlib / Seaborn) for the notebook and the presentation.

Every function returns a Matplotlib figure and saves a PNG to reports/figures/.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src import analysis as A
from src import config as C

K = C.COLORS


def use_style():
    """One dark theme for every chart (matches the dashboard)."""
    plt.rcParams.update({
        "figure.facecolor": K["bg"], "axes.facecolor": K["bg"], "savefig.facecolor": K["bg"],
        "axes.edgecolor": K["grid"], "axes.labelcolor": K["muted"], "axes.titlecolor": K["text"],
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 24, "axes.axisbelow": True, "axes.labelsize": 10.5, "axes.grid": True, "grid.color": K["grid"],
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "xtick.color": K["muted"], "ytick.color": K["muted"],
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "text.color": K["text"],
        "legend.frameon": False, "legend.fontsize": 9.5, "font.size": 10.5,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


use_style()
GROUP_COLORS = {"Indie": K["indie"], "Non-Indie": K["other"]}
pct = PercentFormatter(decimals=0)
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _subtitle(ax, text):
    ax.text(0, 1.012, text, transform=ax.transAxes, color=K["muted"], fontsize=9.5, va="bottom")


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png")
    return fig


# ================================================================ univariate
def releases_per_year(df):
    t = df.groupby(["release_year", "is_indie"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    years = t.index.astype(str)
    ax.bar(years, t[True], color=K["indie"], label="Indie", width=0.62)
    ax.bar(years, t[False], bottom=t[True], color=K["other"], label="Non-Indie", width=0.62,
           edgecolor=K["bg"], linewidth=2)
    for x, total in zip(years, t.sum(axis=1)):
        ax.text(x, total * 1.02, f"{total:,}", ha="center", color=K["text"], fontsize=9.5)
    ax.yaxis.set_major_formatter(thousands)
    ax.set_title("Steam releases keep growing — and most are Indie")
    _subtitle(ax, "Games released per year (2024 used only as a robustness check)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return _save(fig, "01_releases_per_year")


def review_distribution(df):
    s = A.study(df)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.log10(s.loc[~s["few_reviews"], "review_count"])
    ax.hist(x, bins=60, color=K["single"], edgecolor=K["bg"], linewidth=0.6)
    thr = np.log10(np.median(list(A.overview_numbers(df)["hit_threshold_by_year"].values())))
    ax.axvline(thr, color=K["text"], ls="--", lw=1.2)
    ax.text(thr + 0.05, ax.get_ylim()[1] * 0.9, "Hit line ≈ top 5%\nof each release year",
            color=K["text"], fontsize=9.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:,.0f}"))
    ax.set_xlabel("Number of Steam reviews (log scale)")
    ax.set_ylabel("Games")
    few = s["few_reviews"].mean() * 100
    ax.set_title("Most games are barely noticed")
    _subtitle(ax, f"Games with 10+ reviews, 2020–2023  ·  {few:.0f}% of all games have fewer "
                  "than 10 reviews (not shown)")
    return _save(fig, "02_review_distribution")


def rating_distribution(df):
    s = A.study(df)
    s = s[s["review_count"] >= C.MIN_REVIEWS_RATING]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(s["rating_pct"], bins=40, color=K["single"], edgecolor=K["bg"], linewidth=0.6)
    med = s["rating_pct"].median()
    ax.axvline(80, color=K["text"], ls="--", lw=1.2)
    ax.text(79, ax.get_ylim()[1] * 0.9, "80% = Steam\n'Very Positive'", ha="right",
            color=K["text"], fontsize=9.5)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Positive reviews")
    ax.set_ylabel("Games")
    ax.set_title("Players are generous: ratings skew high")
    _subtitle(ax, f"Games with {C.MIN_REVIEWS_RATING}+ reviews, 2020–2023  ·  median {med:.0f}%")
    return _save(fig, "03_rating_distribution")


def price_by_group(df):
    s = A.study(df)
    s = s[(s["price"] > 0) & (s["price"] <= 70)].copy()
    s["group"] = s["is_indie"].map({True: "Indie", False: "Non-Indie"})
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sns.boxplot(data=s, x="price", y="group", hue="group", palette=GROUP_COLORS, legend=False,
                width=0.5, fliersize=1.5, linewidth=1.1, ax=ax,
                flierprops={"markerfacecolor": K["muted"], "markeredgecolor": "none", "alpha": .4},
                medianprops={"color": K["text"], "linewidth": 2},
                whiskerprops={"color": K["muted"]}, capprops={"color": K["muted"]},
                boxprops={"edgecolor": K["bg"]})
    med = s.groupby("group")["price"].median()
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Indie games are priced lower")
    _subtitle(ax, f"Paid games ≤ $70, 2020–2023  ·  median Indie ${med['Indie']:.2f} vs "
                  f"Non-Indie ${med['Non-Indie']:.2f}")
    return _save(fig, "04_price_by_group")


def genre_counts(df):
    s = A.explode_labels(A.study(df), "genres_core")
    t = s.groupby(["genres_core", "is_indie"]).size().unstack(fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t[True], height=0.38, color=K["indie"], label="Indie")
    ax.barh(y - 0.2, t[False], height=0.38, color=K["other"], label="Non-Indie")
    ax.set_yticks(y, t.index)
    ax.xaxis.set_major_formatter(thousands)
    ax.set_xlabel("Games (a game can have several genres)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Casual, Action and Adventure are the most crowded genres")
    _subtitle(ax, "Steam genres, games released 2020–2023")
    return _save(fig, "05_genre_counts")


# ================================================================ bivariate / multivariate
def indie_gap(df):
    t = A.indie_gap(df)
    defs = t["definition"].unique()[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(defs))
    for offset, group in [(0.19, "Indie"), (-0.19, "Non-Indie")]:
        g = t[t["group"] == group].set_index("definition").loc[defs]
        ax.barh(y + offset, g["hit_rate"], height=0.36, color=GROUP_COLORS[group], label=group)
        ax.errorbar(g["hit_rate"], y + offset,
                    xerr=[g["hit_rate"] - g["ci_low"], g["ci_high"] - g["hit_rate"]],
                    fmt="none", ecolor=K["text"], elinewidth=1, capsize=2.5)
        for yy, v, hi in zip(y + offset, g["hit_rate"], g["ci_high"]):
            ax.text(hi + 0.25, yy, f"{v:.1f}%", va="center", color=K["text"], fontsize=9)
    ax.set_yticks(y, defs)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of games that became hits (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    main = t[t["definition"].str.contains("main")].set_index("group")["hit_rate"]
    ax.set_title(f"Indie games are ~{(1 - main['Indie'] / main['Non-Indie']) * 100:.0f}% less "
                 "likely to become hits")
    _subtitle(ax, "The gap holds for every hit definition we tested")
    return _save(fig, "06_indie_gap")


def opportunity_map(df, col="subgenres", min_n=80):
    t, base = A.opportunity_table(df, col=col, min_n=min_n)
    colors = t["signal"].map({"Above average": K["other"], "Below average": K["warn"],
                              "Not clearly different": K["neutral"]})
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.scatter(t["games"], t["hit_rate"], s=t["hits"] * 3 + 25, c=colors, alpha=0.9,
               edgecolors=K["bg"], linewidths=2, zorder=3)
    ax.axhline(base, color=K["muted"], ls="--", lw=1)
    ax.text(t["games"].max() * 1.05, base + 0.3, f"Indie average {base:.1f}%",
            color=K["muted"], fontsize=9, ha="right")
    nudge = {"Open World": (-14, 10), "Sandbox": (8, -12), "Management": (-70, 8),
             "Deckbuilding": (-72, -4), "Farming Sim": (6, 6), "Survival": (8, -10),
             "Roguelite": (-58, 8), "Racing": (-40, -4), "Sports": (-40, 2),
             "Hidden Object": (8, -8), "Idler": (-30, 4),
             "Dating Sim": (-60, 5), "Souls-like": (-56, -12)}
    for _, r in t.iterrows():
        if r["signal"] != "Not clearly different" or r["games"] > 2500:
            ax.annotate(r["label"], (r["games"], r["hit_rate"]),
                        xytext=nudge.get(r["label"], (7, 4)),
                        textcoords="offset points", fontsize=8.5, color=K["text"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(pct)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_xlabel("Competition: number of Indie games with this sub-genre (log scale)")
    ax.set_ylabel("Indie hit rate")
    for label, color in [("Above Indie average", K["other"]), ("Below Indie average", K["warn"]),
                         ("Not clearly different", K["neutral"])]:
        ax.scatter([], [], color=color, s=60, label=label)
    ax.legend(loc="upper right", title="95% CI vs average", title_fontsize=9)
    ax.set_title("Opportunity map: where Indie games break through")
    _subtitle(ax, "Each bubble is a sub-genre (player tags) · bubble size = number of hits · "
                  "Indie games 2020–2023")
    return _save(fig, "07_opportunity_map")


def popularity_vs_satisfaction(df, sample=7000):
    s = A.study(df)
    s = s[s["quadrant"] != "Not enough reviews"]
    pts = s.sample(min(sample, len(s)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    for q, color in C.QUADRANT_COLORS.items():
        p = pts[pts["quadrant"] == q]
        ax.scatter(p["visibility_pctile"], p["rating_pct"], s=9, color=color, alpha=0.55,
                   linewidths=0, label=q)
    ax.axvline(95, color=K["text"], ls="--", lw=1)
    ax.axhline(80, color=K["text"], ls="--", lw=1)
    counts = s["quadrant"].value_counts()
    x0 = s["visibility_pctile"].min() - 1
    spots = {"Popular & Loved": (99.8, 102, "right"), "Popular but Divisive": (99.8, 2, "right"),
             "Hidden Gem": (x0 + 1, 102, "left"), "Low Traction": (x0 + 1, 2, "left")}
    for q, (x, y, ha) in spots.items():
        ax.text(x, y, f"{q}: {counts.get(q, 0):,} games", ha=ha, va="bottom",
                color=K["text"], fontsize=9.5, fontweight="bold")
    ax.set_xlim(x0, 100.5)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Visibility: review-count percentile within release year")
    ax.set_ylabel("Positive reviews (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.06), markerscale=2.5, ncol=2)
    ax.set_title("Being loved is not the same as being seen")
    _subtitle(ax, f"Games with {C.MIN_REVIEWS_RATING}+ reviews, 2020–2023 · "
                  f"random sample of {len(pts):,} points shown")
    return _save(fig, "08_popularity_vs_satisfaction")


def feature_effects(df):
    t = A.feature_effects(df)
    base = A.study(df).query("is_indie")["is_hit"].mean() * 100
    factors = t["factor"].unique()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, factor in zip(axes.flat, factors):
        g = t[t["factor"] == factor]
        ax.bar(g["level"], g["hit_rate"], color=K["single"], width=0.6)
        ax.errorbar(g["level"], g["hit_rate"],
                    yerr=[g["hit_rate"] - g["ci_low"], g["ci_high"] - g["hit_rate"]],
                    fmt="none", ecolor=K["text"], elinewidth=1, capsize=2.5)
        ax.axhline(base, color=K["muted"], ls="--", lw=1)
        ax.set_title(factor, fontsize=11.5)
        ax.yaxis.set_major_formatter(pct)
        ax.tick_params(axis="x", labelsize=8.2, rotation=20)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[len(factors):]:
        ax.axis("off")
    last = list(axes.flat)[-1]
    last.text(0, 0.6, "Dashed line = Indie average\nError bars = 95% CI\n\n"
                      "These are associations,\nnot causes: successful games\noften add "
                      "languages and\nachievements after launch.",
              color=K["muted"], fontsize=9.5, va="center")
    fig.suptitle("What do Indie hits have in common?", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=K["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "09_feature_effects")


def correlation_heatmap(df):
    corr = A.correlation_matrix(df)
    cmap = LinearSegmentedColormap.from_list("div", [K["warn"], K["grid"], K["single"]])
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    corr = corr.iloc[1:, :-1]
    mask = mask[1:, :-1]
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
                linewidths=2, linecolor=K["bg"], annot_kws={"size": 8.5, "color": K["text"]},
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.grid(False)
    ax.set_title("How product features move together")
    _subtitle(ax, "Spearman correlation, games 2020–2023 · review count excluded "
                  "(the hit label is built from it)")
    return _save(fig, "10_correlation_heatmap")


# ================================================================ predictive model (bonus)
def learning_curve(table):
    """Train vs cross-validation ROC-AUC for growing training sizes (src.model)."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for col, std, label, color in [("train_auc", "train_std", "Training", K["indie"]),
                                   ("cv_auc", "cv_std", "Cross-validation", K["single"])]:
        ax.plot(table["train_size"], table[col], marker="o", color=color, label=label, lw=2)
        ax.fill_between(table["train_size"], table[col] - table[std], table[col] + table[std],
                        color=color, alpha=0.15, lw=0)
    ax.xaxis.set_major_formatter(thousands)
    ax.set_xlabel("Training games")
    ax.set_ylabel("ROC-AUC")
    ax.legend(loc="lower right")
    ax.set_title("Learning curve: no sign of overfitting")
    _subtitle(ax, "Logistic regression · 5-fold CV on the training set · band = ±1 std")
    return _save(fig, "11_learning_curve")


def model_coefficients(coefs, n=14):
    """Largest logistic-regression coefficients (log-odds; price and platforms are standardised,
    the other features are 0/1)."""
    t = coefs.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = [K["other"] if c > 0 else K["warn"] for c in t["coef"]]
    ax.barh(t["feature"], t["coef"], color=colors, height=0.65)
    ax.axvline(0, color=K["muted"], lw=1)
    ax.set_xlabel("Coefficient (log-odds of becoming a hit)")
    ax.set_title("What the Hit Predictor relies on")
    _subtitle(ax, "Green raises, orange lowers the predicted chance · associations, not causes")
    return _save(fig, "12_model_coefficients")


ALL_FIGURES = [releases_per_year, review_distribution, rating_distribution, price_by_group,
               genre_counts, indie_gap, opportunity_map, popularity_vs_satisfaction,
               feature_effects, correlation_heatmap]


def save_all(df):
    for fn in ALL_FIGURES:
        plt.close(fn(df))
    print(f"Saved {len(ALL_FIGURES)} figures -> {C.FIGURES}")


if __name__ == "__main__":
    from src.cleaning import load_clean
    save_all(load_clean())
