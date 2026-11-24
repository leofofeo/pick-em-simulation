"""Interactive interface for the standalone pick'em simulation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cli import parse_ints
from model import Config, analytical, compare_standings, distribution_summary, league_statistics, probability_table, rank_summary, simulate, validate_standings, weekly_summary
from plots import make_plots

st.set_page_config(page_title="Pick'em simulation", layout="wide")
st.title("Pick'em league: skill or chance?")
st.caption("Compare league standings with an independent, equal-skill null model.")


@st.cache_data(max_entries=4, show_spinner="Simulating leagues...")
def run_simulation(config: Config, weeks: tuple[int, ...] | None):
    return simulate(config, weeks)


with st.sidebar.form("settings"):
    st.header("League settings")
    players = st.number_input("Participants", min_value=2, max_value=100, value=12)
    games = st.number_input("Games per participant", min_value=1, max_value=1000, value=163)
    score = st.number_input("Observed winning score", min_value=0, value=88)
    ties = st.number_input("Participants tied at the winning score", min_value=1, value=2)
    probability = st.number_input("Baseline pick probability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    simulations = st.number_input("Simulated leagues", min_value=1, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42)
    alpha = st.number_input("Significance level", min_value=0.001, max_value=0.999, value=0.05, step=0.01, format="%.3f")
    standings_source = st.selectbox("Actual standings", ["Example league", "None", "Upload CSV"])
    uploaded = st.file_uploader("Standings CSV", type="csv")
    weeks_text = st.text_input("Games per week (optional)", help="Comma-separated counts summing to total games.")
    winners_text = st.text_input("Observed weekly winners (optional)", help="Comma-separated cumulative winning scores, one per week.")
    st.form_submit_button("Run simulation")

try:
    config = Config(players, games, score, ties, probability, simulations, seed, alpha)
    weeks = parse_ints(weeks_text) if weeks_text.strip() else None
    observed_winners = parse_ints(winners_text) if winners_text.strip() else None
    if observed_winners is not None and weeks is None:
        raise ValueError("Observed weekly winners require games per week")
    if simulations * players > 10_000_000 or (weeks is not None and simulations * len(weeks) > 10_000_000):
        raise ValueError("Reduce simulated leagues, participants or weeks to keep each simulation array within 10 million values")
    standings = None
    if standings_source == "Example league":
        standings = validate_standings(pd.read_csv(Path(__file__).with_name("example_standings.csv")), config)
    elif standings_source == "Upload CSV":
        if uploaded is None:
            raise ValueError("Upload a complete standings CSV or select another standings option")
        standings = validate_standings(pd.read_csv(uploaded), config)
    simulation = run_simulation(config, tuple(weeks) if weeks is not None else None)
    weekly = weekly_summary(simulation, weeks, config, observed_winners) if weeks is not None else None
except (ValueError, argparse.ArgumentTypeError, OSError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

exact, _ = analytical(config)
with st.container(horizontal=True):
    st.metric("Expected individual score", f"{config.games * config.probability:.2f}")
    st.metric("Expected winning score", f"{exact['expected_winner']:.2f}")
    st.metric("Observed winning score", str(config.observed_score))
    st.metric("Winner upper-tail p-value", f"{exact['at_least_one']:.3%}")

if exact["at_least_one"] < config.alpha:
    st.info("The observed winner is unusually high under this model at the selected significance level.")
else:
    st.info("The observed winner is not unusually high under this model at the selected significance level. This does not prove that participants lack skill.")
st.caption("The upper-tail test accounts for all participants. The tie-specific event is descriptive, not the primary hypothesis test.")

st.header("Probability comparison")
st.dataframe(probability_table(config, simulation.ranked), hide_index=True)
st.header("Standings by rank")
st.caption(f"The shaded bands and bounds are {100 * (1 - config.alpha):g}% prediction intervals for each rank, not confidence intervals.")
st.dataframe(rank_summary(simulation.ranked, config.games, config.alpha), hide_index=True)
if standings is not None:
    st.subheader("Actual standings compared with simulated ranks")
    st.caption("Percentiles are the percentage of simulations with a rank score less than or equal to the actual score.")
    st.dataframe(compare_standings(standings, simulation.ranked, config), hide_index=True)

st.header("League-level distributions")
st.dataframe(distribution_summary(league_statistics(simulation.ranked, config.games), config.alpha), hide_index=True)
if weekly is not None:
    st.header("Weekly progression")
    st.dataframe(weekly, hide_index=True)
    st.caption("Repeatedly inspecting weekly p-values inflates the chance of a false positive; these unadjusted values are descriptive, not a sequential test.")

st.header("Visualizations")
figures = make_plots(config, simulation, standings, weeks, observed_winners)
for figure in figures.values():
    st.pyplot(figure)
    plt.close(figure)

with st.expander("Model assumptions and interpretation"):
    st.markdown(
        "Each participant makes every pick; picks are independent across games and participants, "
        "have a constant probability of success, and are binary (no pushes). "
        "Real participants may make correlated picks on the same games, and spreads may not "
        "produce exactly 50% success. These assumptions can materially change the null distribution. "
        "Rank-specific intervals are not simultaneous intervals for the whole standings."
    )
