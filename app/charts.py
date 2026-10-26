"""Interactive Plotly charts for the dashboard (same numbers as src/analysis.py)."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src import analysis as A
from src import config as C

K = C.COLORS
GROUP_COLORS = {"Indie": K["indie"], "Non-Indie": K["other"]}
SIGNAL_COLORS = {"Above average": K["other"], "Below average": K["warn"],
                 "Not clearly different": K["neutral"]}


PIXEL_FONT = "'Press Start 2P', monospace"   # loaded by the app's CSS; titles only
BODY_FONT = "'Source Sans', 'Source Sans Pro', sans-serif"


def style(fig, height=420, title=None, subtitle=None):
    if title:
        fig.update_layout(title=dict(
            text=title, x=0, xanchor="left",
            font=dict(family=PIXEL_FONT, size=13, color=K["text"]),
            subtitle=dict(text=subtitle or "",
                          font=dict(family=BODY_FONT, size=13, color=K["muted"]))))
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=BODY_FONT, color=K["text"], size=12.5),
        margin=dict(l=4, r=4, t=80 if title else 30, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor=K["panel"], font_color=K["text"], bordercolor=K["grid"]),
    )
    fig.update_xaxes(gridcolor=K["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=K["grid"], zeroline=False)
    return fig


def releases_per_year(df):
    t = (df.assign(group=df["is_indie"].map({True: "Indie", False: "Non-Indie"}))
           .groupby(["release_year", "group"]).size().reset_index(name="games"))
    fig = px.bar(t, x="release_year", y="games", color="group", color_discrete_map=GROUP_COLORS,
                 labels={"release_year": "Release year", "games": "Games", "group": ""})
    fig.update_traces(marker_line_color=K["bg"], marker_line_width=2,
                      hovertemplate="%{x}: %{y:,} games<extra>%{fullData.name}</extra>")
    fig.update_xaxes(type="category")
    return style(fig, title="Releases per year", subtitle="Indie games make up most of Steam")


def indie_gap(df_full):
    t = A.indie_gap(df_full)
    fig = go.Figure()
    for group in ["Indie", "Non-Indie"]:
        g = t[t["group"] == group]
        fig.add_bar(y=g["definition"], x=g["hit_rate"], orientation="h", name=group,
                    marker_color=GROUP_COLORS[group],
                    error_x=dict(type="data", symmetric=False, color=K["text"], thickness=1,
                                 array=g["ci_high"] - g["hit_rate"],
                                 arrayminus=g["hit_rate"] - g["ci_low"]),
                    customdata=np.stack([g["games"], g["hits"], g["ci_low"], g["ci_high"]], -1),
                    hovertemplate=("<b>%{y}</b><br>Hit rate %{x:.1f}% "
                                   "(95% CI %{customdata[2]:.1f}–%{customdata[3]:.1f}%)<br>"
                                   "%{customdata[1]:,} hits of %{customdata[0]:,} games"
                                   "<extra>" + group + "</extra>"))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(ticksuffix="%", title="Share of games that became hits")
    return style(fig, height=430, title="The Indie gap holds under every definition",
                 subtitle="Fixed study data (2020–2023 + 2024 check) — not affected by filters")


def opportunity_map(table, base, label_name):
    t = table.copy()
    # Label only the clear signals and the biggest groups (others: hover to see)
    big = t["games"] >= t["games"].quantile(0.85)
    t["text"] = np.where((t["signal"] != "Not clearly different") | big, t["label"], "")
    fig = px.scatter(t, x="games", y="hit_rate", size="hits", color="signal", text="text",
                     hover_name="label",
                     log_x=True, size_max=42, color_discrete_map=SIGNAL_COLORS,
                     custom_data=["games", "hits", "ci_low", "ci_high"],
                     labels={"games": f"Competition: number of games ({label_name})",
                             "hit_rate": "Hit rate", "signal": ""})
    fig.update_traces(textfont=dict(size=10.5, color=K["text"]),
                      marker=dict(line=dict(color=K["bg"], width=2), opacity=0.9),
                      hovertemplate=("<b>%{hovertext}</b><br>%{customdata[0]:,} games · "
                                     "%{customdata[1]:,} hits<br>Hit rate %{y:.1f}% "
                                     "(95% CI %{customdata[2]:.1f}–%{customdata[3]:.1f}%)"
                                     "<extra></extra>"))
    # Alternate label above / below neighbouring bubbles to limit overlaps
    for trace in fig.data:
        order = np.argsort(np.argsort(np.asarray(trace.x)))
        trace.textposition = ["top center" if i % 2 == 0 else "bottom center" for i in order]
    fig.add_hline(y=base, line_dash="dash", line_color=K["muted"],
                  annotation_text=f"Average {base:.1f}%", annotation_position="bottom left",
                  annotation_font_color=K["muted"])
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(dtick=1, tickformat=",d")
    return style(fig, height=580, title="Opportunity map",
                 subtitle="Top-left = fewer competitors and a higher share of hits")


def popularity_vs_satisfaction(df, sample=6000):
    s = df[df["quadrant"] != "Not enough reviews"]
    if len(s) > sample:
        s = s.sample(sample, random_state=42)
    fig = px.scatter(s, x="visibility_pctile", y="rating_pct", color="quadrant",
                     color_discrete_map=C.QUADRANT_COLORS, opacity=0.6, hover_name="name",
                     custom_data=["review_count", "price"],
                     labels={"visibility_pctile": "Visibility (review-count percentile in its year)",
                             "rating_pct": "Positive reviews (%)", "quadrant": ""})
    fig.update_traces(marker=dict(size=6),
                      hovertemplate=("<b>%{hovertext}</b><br>%{customdata[0]:,} reviews · "
                                     "%{y:.0f}% positive<br>Price $%{customdata[1]:.2f}"
                                     "<extra></extra>"))
    fig.add_vline(x=95, line_dash="dash", line_color=K["text"])
    fig.add_hline(y=80, line_dash="dash", line_color=K["text"])
    return style(fig, height=540, title="Being loved is not the same as being seen",
                 subtitle=f"Games with {C.MIN_REVIEWS_RATING}+ reviews · dashed lines = hit "
                          "line (top 5%) and 80% 'Very Positive'")


def feature_effect(table, factor):
    g = table[table["factor"] == factor]
    fig = go.Figure(go.Bar(
        x=g["level"], y=g["hit_rate"], marker_color=K["single"],
        error_y=dict(type="data", symmetric=False, color=K["text"], thickness=1,
                     array=g["ci_high"] - g["hit_rate"], arrayminus=g["hit_rate"] - g["ci_low"]),
        customdata=np.stack([g["games"], g["hits"]], -1),
        hovertemplate="<b>%{x}</b><br>Hit rate %{y:.1f}%<br>%{customdata[1]:,} hits of "
                      "%{customdata[0]:,} games<extra></extra>"))
    fig.update_yaxes(ticksuffix="%")
    return style(fig, height=330, title=factor)


def rating_histogram(df):
    s = df[df["review_count"] >= C.MIN_REVIEWS_RATING]
    fig = px.histogram(s, x="rating_pct", nbins=40, color_discrete_sequence=[K["single"]],
                       labels={"rating_pct": "Positive reviews (%)"})
    fig.update_traces(marker_line_color=K["bg"], marker_line_width=1,
                      hovertemplate="%{x}%: %{y:,} games<extra></extra>")
    fig.update_yaxes(title="Games")
    return style(fig, height=360, title="Rating distribution",
                 subtitle=f"Games with {C.MIN_REVIEWS_RATING}+ reviews")


def price_box(df):
    s = df[(df["price"] > 0) & (df["price"] <= 70)].assign(
        group=lambda d: d["is_indie"].map({True: "Indie", False: "Non-Indie"}))
    fig = px.box(s, x="price", y="group", color="group", color_discrete_map=GROUP_COLORS,
                 points=False, labels={"price": "Price (USD)", "group": ""})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickprefix="$")
    return style(fig, height=300, title="Price of paid games", subtitle="Games ≤ $70")


def cleaning_funnel(log):
    """Rows left after each cleaning step (log has step, rows_after, removed, reason)."""
    t = log.iloc[::-1]
    text = [f"{a:,}" + (f"  (−{r:,})" if r else "") for a, r in zip(t["rows_after"], t["removed"])]
    fig = go.Figure(go.Bar(
        y=t["step"], x=t["rows_after"], orientation="h", marker_color=K["single"],
        marker_line_color=K["bg"], marker_line_width=1, text=text, textposition="outside",
        textfont=dict(color=K["text"], size=12), cliponaxis=False,
        customdata=np.stack([t["rows_before"], t["removed"], t["reason"]], -1),
        hovertemplate=("<b>%{y}</b><br>%{customdata[0]:,} → %{x:,} rows · "
                       "removed %{customdata[1]:,}<br>%{customdata[2]}<extra></extra>")))
    fig.update_xaxes(tickformat=",d", title="Rows left", range=[0, t["rows_after"].max() * 1.22])
    return style(fig, height=380, title="Cleaning funnel",
                 subtitle="Rows left after each step · removed rows in brackets")


def missing_compare(table):
    """Missing % of key fields: raw file vs clean data (grouped bars)."""
    t = table.iloc[::-1]
    fig = go.Figure()
    for col, name, color in [("raw_missing_pct", "Raw file", K["neutral"]),
                             ("clean_missing_pct", "Clean data", K["single"])]:
        fig.add_bar(y=t["field"], x=t[col], orientation="h", name=name, marker_color=color,
                    text=[f"{v:.1f}%" for v in t[col]], textposition="outside",
                    textfont=dict(color=K["text"], size=11), cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>" + name + ": %{x:.1f}% missing<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(ticksuffix="%", range=[0, max(t["raw_missing_pct"].max(),
                                                   t["clean_missing_pct"].max()) * 1.18])
    return style(fig, height=420, title="Missing values before vs after",
                 subtitle="Share of rows with an empty value, raw file vs clean data")
