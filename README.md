# Pick'em league simulation

A standalone Python 3.11+ project. Copy this entire directory to deploy or run it independently of the parent repository. It includes a command-line report and a Streamlit app, both using the same simulation, analytical calculations and plots.

## Setup

From this directory, create and activate a Python environment, then install:

```sh
python -m pip install -r requirements.txt
```

For example, `python -m venv .venv` and `source .venv/bin/activate` create and activate an environment on macOS/Linux. On Windows use `.venv\Scripts\activate`.

## Run

```sh
python cli.py --standings example_standings.csv --output output
streamlit run streamlit_app.py
```

The CLI prints a probability comparison and writes CSV tables and PNG charts into `output/`. Use `python cli.py --help` for all configurable parameters, including the number of participants, games, simulations, seed, observed winner and tie count, baseline probability and significance level. Omitting `--standings` analyzes the winner without assuming the remaining scores. `example_standings.csv` contains the complete current standings supplied in the specification; it is not synthetic.

To supply another league, provide a complete CSV with `participant,score` columns and one unique participant per row. Scores must be integer values from zero through the game count. The first-place score and number of leaders must match the configured observed score and tie count. Tied scores occupy successive sorted ranks; their percentile uses the inclusive empirical CDF (fraction of simulated scores at that rank less than or equal to the observed score).

For weekly progression, provide comma-separated positive game counts summing to the total. Optionally add one **cumulative** observed winning score per week:

```sh
python cli.py --games 5 --observed-score 4 --observed-ties 2 \
  --weeks 2,3 --weekly-winners 2,4 --output output-weekly
```

The final weekly score must match the observed winner, and each week must be attainable from the prior week. The Streamlit sidebar accepts the same inputs. Without observed weekly scores, only the simulated progression and intervals are displayed.

## Interpretation

Under the null model every participant picks every game, all picks are independent across games **and participants**, each pick has a constant success probability, and outcomes are binary (no pushes). Real picks can be correlated because participants choose from the same games; spreads need not yield exactly 50% correctness. These assumptions can substantially affect results.

The primary p-value is the exact probability that **at least one** participant reaches or exceeds the observed winning score. A large p-value does not prove the absence of skill; a small one only indicates evidence against this particular null model. Other events, including the exact tie, are descriptive. Simulated probabilities have Monte Carlo error, especially for rare events. Rank-specific prediction intervals describe possible scores, not confidence intervals for their means, and are not simultaneous coverage intervals. Repeatedly examining weekly unadjusted p-values inflates false-positive risk; the plots are descriptive, not a sequential significance test.

## Tests

Install pytest in the environment if needed, then run `python -m pytest tests` from this directory. The Streamlit UI test uses Streamlit's headless `AppTest`.
