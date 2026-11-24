import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from model import Config, analytical, compare_standings, empirical, league_statistics, rank_summary, simulate, validate_standings, weekly_summary


def test_default_probabilities_match_independent_formulas():
    config = Config()
    exact, pmf = analytical(config)
    tail = binom.sf(87, 163, 0.5)
    assert exact["individual_at_least"] == pytest.approx(tail)
    assert exact["at_least_one"] == pytest.approx(1 - (1 - tail) ** 12)
    assert exact["at_least_t"] == pytest.approx(binom.sf(1, 12, tail))
    assert exact["exact_tie_at_score"] == pytest.approx(66 * binom.pmf(88, 163, 0.5) ** 2 * binom.cdf(87, 163, 0.5) ** 10)
    assert pmf["probability"].sum() == pytest.approx(1)
    assert exact["expected_winner"] == pytest.approx(np.dot(pmf["score"], pmf["probability"]))


def test_exact_events_against_exhaustive_enumeration():
    config = Config(players=3, games=2, observed_score=1, observed_ties=2, probability=0.5)
    exact, _ = analytical(config)
    outcomes = list(itertools.product(range(3), repeat=3))
    weights = [np.prod(binom.pmf(scores, 2, 0.5)) for scores in outcomes]
    tie_probability = sum(weight for scores, weight in zip(outcomes, weights) if max(scores) == 1 and scores.count(1) == 2)
    any_tie = sum(weight for scores, weight in zip(outcomes, weights) if scores.count(max(scores)) >= 2)
    assert exact["exact_tie_at_score"] == pytest.approx(tie_probability)
    assert exact["any_first_place_tie"] == pytest.approx(any_tie)


def test_simulation_agrees_with_analytical_and_is_seeded():
    config = Config(simulations=100_000)
    ranked = simulate(config).ranked
    assert np.array_equal(ranked, simulate(config).ranked)
    assert ranked.shape == (100_000, 12)
    assert np.all(ranked[:, :-1] >= ranked[:, 1:])
    exact, _ = analytical(config)
    estimates = empirical(ranked, config)
    for key in exact:
        assert estimates[key] == pytest.approx(exact[key], abs=0.06 if key == "expected_winner" else 0.005)
    assert rank_summary(ranked, config.games).shape == (12, 9)
    assert league_statistics(ranked, config.games).shape == (100_000, 9)


def test_weekly_distribution_and_observed_progression():
    config = Config(players=3, games=5, observed_score=4, observed_ties=2, simulations=80_000)
    weekly = simulate(config, [2, 3])
    direct = simulate(config)
    assert weekly.weekly_winners is not None
    assert np.array_equal(weekly.weekly_winners[:, -1], weekly.ranked[:, 0])
    assert abs(weekly.ranked[:, 0].mean() - direct.ranked[:, 0].mean()) < 0.03
    progression = weekly_summary(weekly, [2, 3], config, [2, 4])
    assert progression["games"].tolist() == [2, 5]
    assert progression["upper_tail"].iloc[-1] == pytest.approx(analytical(config)[0]["at_least_one"])
    with pytest.raises(ValueError):
        simulate(config, [2, 2])
    with pytest.raises(ValueError):
        weekly_summary(weekly, [2, 3], config, [0, 4])


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_degenerate_probabilities(p):
    config = Config(players=2, games=2, observed_score=2, observed_ties=2, probability=p, simulations=100)
    exact, pmf = analytical(config)
    assert pmf["probability"].sum() == pytest.approx(1)
    assert exact["any_first_place_tie"] == pytest.approx(1)
    assert empirical(simulate(config).ranked, config)["at_least_one"] == pytest.approx(exact["at_least_one"])


def test_standings_validation_and_comparison():
    config = Config(simulations=1000)
    standings = pd.read_csv(Path(__file__).resolve().parents[1] / "example_standings.csv")
    comparison = compare_standings(standings, simulate(config).ranked, config)
    assert comparison["actual_score"].tolist() == [88, 88, 87, 86, 85, 84, 83, 81, 81, 80, 79, 66]
    assert comparison["percentile"].between(0, 100).all()
    assert comparison["outside_interval"].equals(
        (comparison["actual_score"] < comparison["lower"]) | (comparison["actual_score"] > comparison["upper"])
    )
    low_standings = standings.copy()
    low_standings.loc[11, "score"] = 0
    assert compare_standings(low_standings, simulate(config).ranked, config)["outside_interval"].iloc[-1]
    with pytest.raises(ValueError, match="exactly 12"):
        validate_standings(standings.iloc[:2], config)
    invalid = standings.copy()
    invalid["score"] = invalid["score"].astype(float)
    invalid.loc[0, "score"] = 88.5
    with pytest.raises(ValueError, match="integers"):
        validate_standings(invalid, config)
    invalid = standings.copy()
    invalid.loc[0, "participant"] = invalid.loc[1, "participant"]
    with pytest.raises(ValueError, match="unique"):
        validate_standings(invalid, config)
    with pytest.raises(ValueError, match="tie count"):
        validate_standings(standings, Config(observed_ties=1))


@pytest.mark.parametrize("kwargs", [
    {"games": 0}, {"players": 1}, {"probability": -0.1}, {"alpha": 1},
    {"observed_score": 164}, {"observed_ties": 13}, {"simulations": 0}, {"seed": -1},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
