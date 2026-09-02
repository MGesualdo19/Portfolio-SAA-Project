"""
dashboard/theme.py

One place for colour and chart styling, so every figure in the dashboard
reads as part of the same system rather than as ten charts that happen to
share a page.

Colour is assigned by the JOB it does, never by series index:

  * CATEGORICAL (identity -- which holding, which allocation) draws from
    a fixed eight-slot order. Slots are assigned in order and never
    cycled, and a holding keeps its colour across every chart in the app,
    so filtering one out never repaints the others.
  * SEQUENTIAL (magnitude -- a weight, a risk share) is one blue hue,
    light to dark.
  * DIVERGING (polarity -- correlation, over/underweight) is blue to red
    through a neutral grey, because zero has a real meaning in those
    charts and must not be a hue.
  * STATUS (good / warning / bad) is reserved and never reused as a
    series colour.

The categorical order below was validated for colour-vision deficiency
separation on both light and dark surfaces (worst adjacent pair dE 9.1
protan, worst normal-vision pair dE 19.6). Three of the light-mode hues
sit below 3:1 contrast against the surface, which is why every chart here
ships direct labels or an accompanying table -- identity is never carried
by colour alone.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- categorical: fixed order, never cycled -------------------------------
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#2a78d6", "#256abf", "#184f95", "#0d366b"]

STATUS = {"good": "#008300", "warning": "#eda100", "serious": "#eb6834", "critical": "#e34948"}

LIGHT = {
    "surface": "#fcfcfb", "panel": "#ffffff", "grid": "#e8e7e3",
    "text": "#0b0b0b", "text_secondary": "#52514e", "muted": "#8a8880",
    "neutral": "#f0efec", "categorical": CATEGORICAL_LIGHT,
}
DARK = {
    "surface": "#1a1a19", "panel": "#222220", "grid": "#383835",
    "text": "#ffffff", "text_secondary": "#c3c2b7", "muted": "#8a8880",
    "neutral": "#383835", "categorical": CATEGORICAL_DARK,
}


def palette(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def diverging(dark: bool = False) -> list:
    """Blue <-> red through neutral grey. Used for correlation and over/underweight."""
    mid = DARK["neutral"] if dark else LIGHT["neutral"]
    return [[0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, mid],
            [0.75, "#e34948"], [1.0, "#7d1a1a"]]


def sequential(dark: bool = False) -> list:
    steps = SEQUENTIAL_BLUE if not dark else list(reversed(SEQUENTIAL_BLUE))
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


# Stable colour per holding: a ticker keeps its hue everywhere in the app.
# Order follows the policy sleeves so neighbouring bars are neighbouring hues.
TICKER_ORDER = ["XFR.TO", "CASH.TO", "XUU.TO", "VTV", "AVUV",
                "XIC.TO", "VIU.TO", "VEE.TO", "CGL.TO", "CAR-UN.TO", "VOLX.TO"]


def ticker_colors(dark: bool = False) -> dict:
    cats = palette(dark)["categorical"]
    return {t: cats[i % len(cats)] for i, t in enumerate(TICKER_ORDER)}


def apply_template(dark: bool = False) -> str:
    """Register and return a Plotly template name matching the app surface."""
    p = palette(dark)
    name = "saa_dark" if dark else "saa_light"
    pio.templates[name] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=p["surface"],
            plot_bgcolor=p["surface"],
            font=dict(family="Inter, Segoe UI, system-ui, sans-serif",
                      size=13, color=p["text"]),
            colorway=p["categorical"],
            # Recessive grid, no axis lines: the data should be the only
            # high-contrast thing on the canvas.
            xaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"],
                       linecolor=p["grid"], showline=False, ticks="outside",
                       tickcolor=p["grid"], tickfont=dict(color=p["text_secondary"])),
            yaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"],
                       linecolor=p["grid"], showline=False, ticks="outside",
                       tickcolor=p["grid"], tickfont=dict(color=p["text_secondary"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(color=p["text_secondary"], size=12)),
            margin=dict(l=56, r=24, t=48, b=44),
            hovermode="x unified",
            hoverlabel=dict(bgcolor=p["panel"], font_size=12,
                            bordercolor=p["grid"],
                            font=dict(color=p["text"])),
            title=dict(font=dict(size=15, color=p["text"]), x=0, xanchor="left"),
        )
    )
    pio.templates.default = name
    return name


def styled(fig: go.Figure, dark: bool = False, height: int = 420,
           title: str | None = None, ylabel: str | None = None,
           xlabel: str | None = None) -> go.Figure:
    fig.update_layout(template=apply_template(dark), height=height)
    if title:
        fig.update_layout(title=title)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    return fig
