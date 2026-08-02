#!/usr/bin/env python3
"""
Draw Figure 3: global vs subject-wise uncertainty confidence intervals.

Usage:
  python scripts/plot_uncertainty_confidence_intervals.py
  python scripts/plot_uncertainty_confidence_intervals.py --csv path/to/similarities.csv

The optional CSV should contain:
  subject, similarity

Optional columns:
  sample_id, matched
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


def make_demo_data(seed: int = 7, n_per_subject: int = 130) -> pd.DataFrame:
    """Create an illustrative distribution that mirrors cross-subject EEG quality gaps."""
    rng = np.random.default_rng(seed)

    # Higher mean/lower std means cleaner alignment; lower mean/higher std means noisier EEG.
    params = {
        "sub-01": (0.205, 0.052),
        "sub-02": (0.255, 0.035),
        "sub-03": (0.265, 0.033),
        "sub-04": (0.252, 0.037),
        "sub-05": (0.215, 0.058),
        "sub-06": (0.305, 0.030),
        "sub-07": (0.258, 0.036),
        "sub-08": (0.315, 0.028),
        "sub-09": (0.225, 0.055),
        "sub-10": (0.325, 0.027),
    }

    rows = []
    for subject, (mu, sigma) in params.items():
        values = rng.normal(mu, sigma, n_per_subject)
        values = np.clip(values, 0.02, 0.48)
        rows.extend({"subject": subject, "similarity": value} for value in values)
    return pd.DataFrame(rows)


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path is None:
        return make_demo_data()

    df = pd.read_csv(csv_path)
    required = {"subject", "similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["similarity"] = pd.to_numeric(df["similarity"], errors="raise")
    return df


def confidence_bounds(values: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = float(norm.ppf(1 - alpha / 2))
    return mu, sigma, mu - z * sigma, mu + z * sigma


def scatter_subjects(ax: plt.Axes, df: pd.DataFrame, subjects: list[str]) -> None:
    palette = plt.get_cmap("tab10")
    rng = np.random.default_rng(23)

    for i, subject in enumerate(subjects, start=1):
        values = df.loc[df["subject"] == subject, "similarity"].to_numpy()
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=15,
            alpha=0.56,
            color=palette((i - 1) % 10),
            linewidths=0,
        )


def plot_global_panel(ax: plt.Axes, df: pd.DataFrame, subjects: list[str], alpha: float) -> None:
    scatter_subjects(ax, df, subjects)
    _, _, lower, upper = confidence_bounds(df["similarity"].to_numpy(), alpha)

    ax.axhspan(lower, upper, color="#9ecae1", alpha=0.20, zorder=0)
    ax.axhline(lower, color="#1f77b4", lw=2.0, ls="--", label="Global CI")
    ax.axhline(upper, color="#1f77b4", lw=2.0, ls="--")

    ax.text(
        len(subjects) + 0.35,
        upper,
        "global upper",
        va="center",
        ha="left",
        fontsize=9,
        color="#1f4e79",
    )
    ax.text(
        len(subjects) + 0.35,
        lower,
        "global lower",
        va="center",
        ha="left",
        fontsize=9,
        color="#1f4e79",
    )
    ax.set_title("(a) Original global uncertainty interval", loc="left", weight="bold")


def plot_subjectwise_panel(ax: plt.Axes, df: pd.DataFrame, subjects: list[str], alpha: float) -> None:
    scatter_subjects(ax, df, subjects)

    for i, subject in enumerate(subjects, start=1):
        values = df.loc[df["subject"] == subject, "similarity"].to_numpy()
        _, _, lower, upper = confidence_bounds(values, alpha)
        ax.vlines(i, lower, upper, color="#2ca25f", lw=5.0, alpha=0.55)
        ax.hlines([lower, upper], i - 0.28, i + 0.28, color="#006d2c", lw=1.8)

    ax.plot([], [], color="#2ca25f", lw=5.0, alpha=0.55, label="Subject-wise CI")
    ax.set_title("(b) Proposed subject-wise uncertainty intervals", loc="left", weight="bold")


def style_axis(ax: plt.Axes, subjects: list[str]) -> None:
    ax.set_xlim(0.35, len(subjects) + 1.15)
    ax.set_xticks(range(1, len(subjects) + 1))
    ax.set_xticklabels(subjects, rotation=35, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Paired EEG-image semantic similarity")
    ax.grid(axis="y", color="#d9d9d9", lw=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="CSV with columns: subject, similarity")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha")
    parser.add_argument(
        "--out",
        default="assets/figure3_uncertainty_confidence_intervals.png",
        help="Output image path",
    )
    parser.add_argument("--pdf", action="store_true", help="Also export a PDF beside the PNG")
    args = parser.parse_args()

    df = load_data(args.csv)
    subjects = [s for s in SUBJECTS if s in set(df["subject"])]
    if not subjects:
        subjects = sorted(df["subject"].unique())

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharey=True)
    plot_global_panel(axes[0], df, subjects, args.alpha)
    plot_subjectwise_panel(axes[1], df, subjects, args.alpha)

    for ax in axes:
        style_axis(ax, subjects)

    fig.suptitle(
        "Subject-specific calibration reduces cross-subject uncertainty bias",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")

    if args.pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
