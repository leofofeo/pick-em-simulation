"""Command-line entry point for the pick'em simulation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from model import Config, analytical, compare_standings, distribution_summary, league_statistics, probability_table, rank_summary, simulate, validate_standings, weekly_summary
from plots import make_plots


def parse_ints(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    try:
        return [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate an independent-pick NFL pick'em league")
    parser.add_argument("--players", type=int, default=12)
    parser.add_argument("--games", type=int, default=163)
    parser.add_argument("--observed-score", type=int, default=88)
    parser.add_argument("--observed-ties", type=int, default=2)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--standings", type=Path, help="Complete CSV with participant and score columns")
    parser.add_argument("--weeks", type=parse_ints, help="Comma-separated games per week; must sum to --games")
    parser.add_argument("--weekly-winners", type=parse_ints, help="Comma-separated observed cumulative winning scores")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    try:
        config = Config(args.players, args.games, args.observed_score, args.observed_ties, args.probability, args.simulations, args.seed, args.alpha)
        if args.weekly_winners is not None and args.weeks is None:
            raise ValueError("--weekly-winners requires --weeks")
        standings = validate_standings(pd.read_csv(args.standings), config) if args.standings else None
        simulation = simulate(config, args.weeks)
        weekly = weekly_summary(simulation, args.weeks, config, args.weekly_winners) if args.weeks else None
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    ranks = rank_summary(simulation.ranked, config.games, config.alpha)
    probabilities = probability_table(config, simulation.ranked)
    exact, maximum = analytical(config)
    ranks.to_csv(args.output / "rank_summary.csv", index=False)
    probabilities.to_csv(args.output / "probabilities.csv", index=False)
    distribution_summary(league_statistics(simulation.ranked, config.games), config.alpha).to_csv(args.output / "league_summary.csv", index=False)
    maximum.to_csv(args.output / "winning_score_pmf.csv", index=False)
    if standings is not None:
        compare_standings(standings, simulation.ranked, config).to_csv(args.output / "standings_comparison.csv", index=False)
    if weekly is not None:
        weekly.to_csv(args.output / "weekly_summary.csv", index=False)
    figures = make_plots(config, simulation, standings, args.weeks, args.weekly_winners)
    for name, figure in figures.items():
        figure.savefig(args.output / f"{name}.png", dpi=150)
        plt.close(figure)
    print(probabilities.to_string(index=False))
    print(f"\nExpected winning score: analytical {exact['expected_winner']:.3f}; simulated {simulation.ranked[:, 0].mean():.3f}")
    print(f"Primary upper-tail p-value: {exact['at_least_one']:.6f} (alpha={config.alpha:g})")
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
