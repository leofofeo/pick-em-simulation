"""Matplotlib figures shared by the command line and Streamlit interfaces."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import binom

from model import Config, Simulation, analytical, league_statistics, rank_summary, weekly_summary


def make_plots(
    config: Config,
    simulation: Simulation,
    standings: pd.DataFrame | None = None,
    weeks: list[int] | None = None,
    observed_winners: list[int] | None = None,
) -> dict[str, Figure]:
    """Create figures without saving them, so callers control output and lifetime."""
    ranked = simulation.ranked
    stats = league_statistics(ranked, config.games)
    ranks = rank_summary(ranked, config.games, config.alpha)
    _, maximum = analytical(config)
    interval_label = f"{100 * (1 - config.alpha):g}% prediction interval"
    figures = {}

    fig, ax = plt.subplots()
    scores = np.arange(config.games + 1)
    ax.plot(scores, binom.pmf(scores, config.games, config.probability), label="Individual score PMF")
    ax.axvline(config.games * config.probability, color="tab:green", linestyle="--", label="Expected individual")
    ax.axvline(config.observed_score, color="tab:red", linestyle="--", label="Observed winner")
    ax.set(xlabel="Score", ylabel="Probability", title="Individual score distribution")
    individual_bounds = binom.ppf([0.0001, 0.9999], config.games, config.probability)
    ax.set_xlim(min(individual_bounds[0], config.observed_score) - 2, max(individual_bounds[1], config.observed_score) + 2)
    ax.legend()
    figures["individual_scores"] = fig

    fig, ax = plt.subplots()
    counts = np.bincount(ranked[:, 0], minlength=config.games + 1) / config.simulations
    ax.bar(scores, counts, width=0.9, alpha=0.5, label="Simulated")
    ax.plot(maximum["score"], maximum["probability"], color="tab:orange", label="Analytical PMF")
    ax.axvline(ranked[:, 0].mean(), color="tab:green", linestyle="--", label="Simulated mean")
    ax.axvline(config.observed_score, color="tab:red", linestyle="--", label="Observed winner")
    low, high = np.quantile(ranked[:, 0], [config.alpha / 2, 1 - config.alpha / 2])
    ax.axvspan(low, high, color="tab:blue", alpha=0.12, label=interval_label)
    ax.set(xlabel="Winning score", ylabel="Probability", title="Winning score distribution")
    visible_scores = scores[(counts > 0) | (maximum["probability"].to_numpy() > 0.00001)]
    ax.set_xlim(min(visible_scores[0], config.observed_score) - 2, max(visible_scores[-1], config.observed_score) + 2)
    ax.legend()
    figures["winning_scores"] = fig

    fig, ax = plt.subplots()
    ax.fill_between(ranks["rank"], ranks["lower"], ranks["upper"], alpha=0.2, label=interval_label)
    ax.plot(ranks["rank"], ranks["mean"], marker="o", label="Expected")
    if standings is not None:
        ax.plot(ranks["rank"], standings["score"], marker="o", label="Actual")
    ax.set(xlabel="Rank", ylabel="Score", title="Standings by rank", xticks=ranks["rank"])
    ax.legend()
    figures["standings"] = fig

    fig, ax = plt.subplots()
    gaps = stats["first_to_last"].to_numpy()
    gap_counts = np.bincount(gaps, minlength=config.games + 1) / config.simulations
    ax.bar(np.arange(len(gap_counts)), gap_counts, width=0.9, alpha=0.7)
    if standings is not None:
        gap = int(standings["score"].max() - standings["score"].min())
        ax.axvline(gap, color="tab:red", linestyle="--", label="Observed gap")
        ax.legend()
    else:
        gap = 0
    ax.set_xlim(-1, max(int(gaps.max()), gap) + 2)
    ax.set(xlabel="First-to-last score difference", ylabel="Probability", title="League score spread")
    figures["score_spread"] = fig

    fig, ax = plt.subplots()
    ax.hist(stats["accuracy"], bins=30, weights=np.ones(len(stats)) / len(stats), alpha=0.7)
    if standings is not None:
        ax.axvline(standings["score"].mean() / config.games, color="tab:red", linestyle="--", label="Observed accuracy")
        ax.legend()
    ax.set(xlabel="Overall picking accuracy", ylabel="Probability per bin", title="League-wide accuracy")
    figures["league_accuracy"] = fig

    if weeks is not None:
        weekly = weekly_summary(simulation, weeks, config, observed_winners)
        fig, ax = plt.subplots()
        ax.fill_between(weekly["week"], weekly["lower"], weekly["upper"], alpha=0.2, label=interval_label)
        ax.plot(weekly["week"], weekly["expected_winner"], marker="o", label="Expected winner")
        if observed_winners is not None:
            ax.plot(weekly["week"], weekly["actual_winner"], marker="o", label="Observed winner")
        ax.set(xlabel="Week", ylabel="Cumulative score", title="Weekly winning scores")
        ax.legend()
        figures["weekly_scores"] = fig
        if observed_winners is not None:
            fig, ax = plt.subplots()
            ax.plot(weekly["week"], weekly["upper_tail"], marker="o")
            ax.axhline(config.alpha, color="tab:red", linestyle="--", label=f"Alpha = {config.alpha:g}")
            ax.set(xlabel="Week", ylabel="Upper-tail probability", title="Weekly winner upper-tail probabilities", ylim=(0, 1))
            ax.legend()
            figures["weekly_probabilities"] = fig
    for fig in figures.values():
        fig.tight_layout()
    return figures
