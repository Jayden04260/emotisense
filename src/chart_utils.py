"""
Shared brand palette + chart helper for EmotiSense.

Used by app/app.py (the per-prediction confidence chart) and by the
model-comparison scripts (compare_text_models.py, compare_audio_models.py)
so every chart in the project follows the same validated, colorblind-safe
"emphasis" pattern instead of each script inventing its own colors:
one accent hue on the value that matters, neutral gray on the rest.

See the dataviz color-formula notes inline in render_emphasis_bar_chart.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Brand palette (validated categorical slot 1, light mode - see palette.md
# in the dataviz skill this was built against). Reused everywhere so the
# whole project reads as one system rather than a chart-per-script grab bag.
# --------------------------------------------------------------------------

ACCENT = "#2a78d6"
ACCENT_DARK = "#184f95"
MUTED_BAR = "#c3c2b7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def render_emphasis_bar_chart(items, highlight_label, xlabel, value_scale=100.0, value_suffix="%"):
    """Horizontal bar chart using the "emphasis" pattern: the job here is
    comparing magnitude across nominal categories (which emotion, which
    model), so per the dataviz color-formula that's a sequential/emphasis
    color job, not a rainbow of categorical hues - one accent color marks
    the value that matters (the prediction, the best model), everything
    else recedes to a neutral gray so it stays legible for CVD and
    normal vision alike without needing a legend.

    items: iterable of (label, raw_value) pairs, e.g. [("joy", 0.68), ...]
           raw_value is in its natural unit (a 0-1 probability, a 0-1
           accuracy, seconds, etc.) - multiply by value_scale to get the
           displayed number (100 for a 0-1 fraction shown as a percentage,
           1 to leave it as-is).
    highlight_label: which item's label gets the accent color.
    xlabel: axis caption, e.g. "Confidence (%)" or "Test accuracy (%)".
    """
    items = sorted(items, key=lambda kv: kv[1], reverse=True)
    labels = [str(lbl) for lbl, _ in items]
    values = [v * value_scale for _, v in items]
    colors = [ACCENT if lbl == highlight_label else MUTED_BAR for lbl, _ in items]

    fig_height = 0.6 * len(items) + 0.8
    fig, ax = plt.subplots(figsize=(6, fig_height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y_pos = np.arange(len(items))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()  # highest value at the top

    max_value = max(values) if values else 1.0
    ax.set_xlim(0, max_value * 1.2)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)

    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    for bar, value, (lbl, _) in zip(bars, values, items):
        is_winner = lbl == highlight_label
        label_color = ACCENT_DARK if is_winner else INK_MUTED
        weight = "bold" if is_winner else "normal"
        ax.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}{value_suffix}",
            va="center",
            ha="left",
            color=label_color,
            fontsize=10,
            fontweight=weight,
        )

    plt.tight_layout()
    return fig


def render_trend_line_chart(y_values, ylabel):
    """Simple line chart for a single series over successive predictions
    (e.g. confidence over time on the Phase 5 dashboard) - same brand
    palette as render_emphasis_bar_chart, so it still reads as part of
    one visual system rather than falling back to matplotlib/Streamlit
    defaults for anything that isn't a magnitude-comparison bar chart."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(1, len(y_values) + 1)
    ax.plot(
        x, y_values, color=ACCENT, linewidth=2, marker="o", markersize=4,
        markerfacecolor=ACCENT, markeredgecolor=SURFACE, zorder=3,
    )

    ax.set_xlabel("Prediction #", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig
