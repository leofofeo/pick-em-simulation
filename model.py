"""Independent-pick null model and league-level summaries."""

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import binom


@dataclass(frozen=True)
class Config:
    """Parameters for the null model and observed first-place result."""

    players: int = 12
    games: int = 163
    observed_score: int = 88
    observed_ties: int = 2
    probability: float = 0.5
    simulations: int = 100_000
    seed: int = 42
    alpha: float = 0.05

    def __post_init__(self) -> None:
        for name in ("players", "games", "observed_score", "observed_ties", "simulations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.players < 2 or self.games < 1 or self.simulations < 1 or self.seed < 0:
            raise ValueError("Use at least two players, one game and one simulation, and a nonnegative seed")
        if not 0 <= self.observed_score <= self.games:
            raise ValueError("Observed score must be between zero and the number of games")
        if not 1 <= self.observed_ties <= self.players:
            raise ValueError("Observed tie count must be between one and the number of players")
        if not 0 <= self.probability <= 1 or not 0 < self.alpha < 1:
            raise ValueError("Probability must be between 0 and 1 and alpha must be between 0 and 1 exclusive")


@dataclass
class Simulation:
    """Ranked final scores and, when requested, weekly winning scores."""

    ranked: np.ndarray
    weekly_winners: np.ndarray | None = None


def validate_weeks(weeks: list[int] | tuple[int, ...], games: int) -> None:
    """Check that positive weekly counts sum to the configured total."""
    if not weeks or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in weeks):
        raise ValueError("Weekly game counts must be positive integers")
    if sum(weeks) != games:
        raise ValueError("Weekly game counts must sum to the total number of games")


def simulate(config: Config, weeks: list[int] | tuple[int, ...] | None = None) -> Simulation:
    """Draw independent binomial participant scores, optionally tracking weekly leaders."""
    rng = np.random.default_rng(config.seed)
    shape = (config.simulations, config.players)
    if weeks is None:
        scores = rng.binomial(config.games, config.probability, size=shape)
        winners = None
    else:
        validate_weeks(weeks, config.games)
        scores = np.zeros(shape, dtype=np.int64)
        winners = np.empty((config.simulations, len(weeks)), dtype=np.int64)
        for index, games in enumerate(weeks):
            scores += rng.binomial(games, config.probability, size=shape)
            winners[:, index] = scores.max(axis=1)
    return Simulation(np.sort(scores, axis=1)[:, ::-1], winners)


def rank_summary(ranked: np.ndarray, games: int, alpha: float = 0.05) -> pd.DataFrame:
    """Summarize each rank with central prediction intervals, not confidence intervals."""
    low, q25, q75, high = np.percentile(ranked, [100 * alpha / 2, 25, 75, 100 * (1 - alpha / 2)], axis=0)
    return pd.DataFrame({
        "rank": np.arange(1, ranked.shape[1] + 1),
        "mean": ranked.mean(axis=0),
        "median": np.median(ranked, axis=0),
        "std": ranked.std(axis=0),
        "mean_accuracy": ranked.mean(axis=0) / games,
        "lower": low, "q25": q25, "q75": q75, "upper": high,
    })


def league_statistics(ranked: np.ndarray, games: int) -> pd.DataFrame:
    """Return one row of league-level statistics per simulated league."""
    return pd.DataFrame({
        "highest": ranked[:, 0],
        "lowest": ranked[:, -1],
        "mean": ranked.mean(axis=1),
        "median": np.median(ranked, axis=1),
        "std": ranked.std(axis=1),
        "first_to_last": ranked[:, 0] - ranked[:, -1],
        "first_to_second": ranked[:, 0] - ranked[:, 1],
        "first_place_ties": (ranked == ranked[:, :1]).sum(axis=1),
        "accuracy": ranked.mean(axis=1) / games,
    })


def distribution_summary(data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Summarize the sampling distribution of every league statistic."""
    return pd.DataFrame({
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(ddof=0),
        "lower": data.quantile(alpha / 2),
        "q25": data.quantile(0.25),
        "q75": data.quantile(0.75),
        "upper": data.quantile(1 - alpha / 2),
    }).rename_axis("statistic").reset_index()


def analytical(config: Config) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute exact null probabilities and the maximum's probability mass function."""
    n, m, p, k, t = config.games, config.players, config.probability, config.observed_score, config.observed_ties
    scores = np.arange(n + 1)
    cdf = binom.cdf(scores, n, p)
    maximum_pmf = np.diff(np.power(cdf, m), prepend=0.0)
    individual_tail = float(binom.sf(k - 1, n, p))
    exact = float(binom.pmf(k, n, p))
    below = float(binom.cdf(k - 1, n, p))
    probabilities = {
        "individual_at_least": individual_tail,
        "at_least_one": float(-np.expm1(m * np.log1p(-individual_tail))) if individual_tail < 1 else 1.0,
        "at_least_t": float(binom.sf(t - 1, m, individual_tail)),
        "nobody_higher": float(binom.cdf(k, n, p) ** m),
        "exact_tie_at_score": float(comb(m, t) * exact**t * below**(m - t)),
        "any_first_place_tie": float(1 - np.sum(m * binom.pmf(scores, n, p) * binom.cdf(scores - 1, n, p)**(m - 1))),
        "winning_score_exact": float(maximum_pmf[k]),
        "expected_winner": float(np.dot(scores, maximum_pmf)),
    }
    return probabilities, pd.DataFrame({"score": scores, "probability": maximum_pmf})


def empirical(ranked: np.ndarray, config: Config) -> dict[str, float]:
    """Estimate the same events as the analytical calculations."""
    winners = ranked[:, 0]
    tied = (ranked == winners[:, None]).sum(axis=1)
    return {
        "individual_at_least": float(np.mean(ranked >= config.observed_score)),
        "at_least_one": float(np.mean(winners >= config.observed_score)),
        "at_least_t": float(np.mean((ranked >= config.observed_score).sum(axis=1) >= config.observed_ties)),
        "nobody_higher": float(np.mean(winners <= config.observed_score)),
        "exact_tie_at_score": float(np.mean((winners == config.observed_score) & (tied == config.observed_ties))),
        "any_first_place_tie": float(np.mean(tied >= 2)),
        "winning_score_exact": float(np.mean(winners == config.observed_score)),
        "expected_winner": float(winners.mean()),
    }


def probability_table(config: Config, ranked: np.ndarray) -> pd.DataFrame:
    """Pair exact and simulated probabilities for the same events."""
    exact, _ = analytical(config)
    simulated = empirical(ranked, config)
    labels = {
        "individual_at_least": f"Individual scores at least {config.observed_score}",
        "at_least_one": f"At least one scores at least {config.observed_score}",
        "at_least_t": f"At least {config.observed_ties} score at least {config.observed_score}",
        "nobody_higher": f"Nobody scores higher than {config.observed_score}",
        "exact_tie_at_score": f"Exactly {config.observed_ties} share first at {config.observed_score}",
        "any_first_place_tie": "At least two share first at any score",
        "winning_score_exact": f"Winning score is exactly {config.observed_score}",
    }
    return pd.DataFrame([{"event": label, "analytical": exact[key], "simulated": simulated[key]} for key, label in labels.items()])


def validate_standings(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Validate a complete standings file and return it sorted by score."""
    if not {"participant", "score"}.issubset(data.columns):
        raise ValueError("Standings CSV requires participant and score columns")
    if len(data) != config.players:
        raise ValueError(f"Standings must contain exactly {config.players} participants")
    names = data["participant"]
    if names.isna().any() or names.astype(str).str.strip().eq("").any() or names.astype(str).str.strip().duplicated().any():
        raise ValueError("Participant identifiers must be nonempty and unique")
    values = pd.to_numeric(data["score"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or not ((values % 1 == 0) & values.between(0, config.games)).all():
        raise ValueError("Scores must be integers between zero and the number of games")
    result = pd.DataFrame({"participant": names.astype(str).str.strip(), "score": values.astype(int)})
    result = result.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    if result["score"].iloc[0] != config.observed_score or (result["score"] == config.observed_score).sum() != config.observed_ties:
        raise ValueError("Standings winner and tie count must match the configured observed score and tie count")
    return result


def compare_standings(standings: pd.DataFrame, ranked: np.ndarray, config: Config) -> pd.DataFrame:
    """Compare actual standings to corresponding simulated ranks using an inclusive ECDF."""
    standings = validate_standings(standings, config)
    summary = rank_summary(ranked, config.games, config.alpha)
    actual = standings["score"].to_numpy()
    return pd.DataFrame({
        "rank": summary["rank"], "participant": standings["participant"],
        "actual_score": actual, "actual_accuracy": actual / config.games,
        "expected_score": summary["mean"], "difference": actual - summary["mean"],
        "lower": summary["lower"], "upper": summary["upper"],
        "percentile": (ranked <= actual).mean(axis=0) * 100,
        "outside_interval": (actual < summary["lower"]) | (actual > summary["upper"]),
    })


def weekly_summary(
    simulation: Simulation, weeks: list[int] | tuple[int, ...], config: Config,
    observed_winners: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize weekly winning-score distributions and optional observed upper tails."""
    validate_weeks(weeks, config.games)
    winners = simulation.weekly_winners
    if winners is None or winners.shape[1] != len(weeks):
        raise ValueError("Weekly simulation results must match the supplied schedule")
    cumulative = np.cumsum(weeks)
    result = pd.DataFrame({
        "week": np.arange(1, len(weeks) + 1), "games": cumulative,
        "expected_winner": winners.mean(axis=0),
        "lower": np.percentile(winners, 100 * config.alpha / 2, axis=0),
        "upper": np.percentile(winners, 100 * (1 - config.alpha / 2), axis=0),
    })
    if observed_winners is not None:
        if len(observed_winners) != len(weeks):
            raise ValueError("Provide one observed cumulative winner per week")
        previous = 0
        for value, count, total in zip(observed_winners, weeks, cumulative):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not previous <= value <= min(total, previous + count):
                raise ValueError("Weekly winners must be attainable nondecreasing cumulative integer scores")
            previous = value
        if observed_winners[-1] != config.observed_score:
            raise ValueError("Final weekly winner must match the configured observed score")
        tails = binom.sf(np.asarray(observed_winners) - 1, cumulative, config.probability)
        result["actual_winner"] = observed_winners
        result["upper_tail"] = binom.sf(0, config.players, tails)
    return result
