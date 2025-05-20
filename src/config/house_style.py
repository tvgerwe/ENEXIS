# src/config/house_style.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

COLORS = {
    "main": "#0077FF",
    "positive": "#3BB273",
    "negative": "#FF5733",
    "neutral": "#2E2E2E",
    "background": "#F6F6F6"
}

FONTS = {
    "title": "Anton",
    "body": "Quicksand"
}

def apply_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": COLORS["background"],
        "font.family": FONTS["body"],
    })

def apply_seaborn_style():
    sns.set_style("whitegrid")
    sns.set_palette([COLORS["main"], COLORS["positive"], COLORS["negative"]])

def style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return df.style.format(precision=4).background_gradient(cmap="Blues")

def get_plotly_layout(title: str):
    return {
        "title": {"text": title, "font": {"family": FONTS["title"], "size": 20}},
        "font": {"family": FONTS["body"]},
        "plot_bgcolor": COLORS["background"],
        "paper_bgcolor": COLORS["background"],
    }
