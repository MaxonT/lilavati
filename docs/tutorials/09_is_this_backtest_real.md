# Tutorial 09: Is This Backtest Real? Deflated Sharpe, PBO and Purged Cross-Validation

**Who this is for:** anyone who selects trading rules or models by backtest, and anyone reviewing a
backtest someone else selected; a session in a course on AI in finance.
**Time:** about 45 minutes. **Cost:** free (no API calls).

## Learning objectives

1. Explain why the best of many backtests overstates its own Sharpe ratio (selection bias).
2. Compute the Deflated Sharpe Ratio (DSR) and say what it needs from you: the number of independent
   trials.
3. Estimate the probability of backtest overfitting (PBO) by combinatorially symmetric
   cross-validation, which needs no count of trials.
4. Build purged and embargoed cross-validation splits for labels that span several bars.
5. State what these statistics do and do not show.

## Where this fits

[Tutorial 02](02_statistical_arbitrage_and_execution_agent.md) builds a trading rule and checks each
number it uses. This tutorial asks the next question: when a rule was chosen as the best of many
backtests, how much of its performance is selection? It needs no model and no API key.

## 1. The idea

Try 400 settings of a trading rule on prices with no structure and the best one will look good: its
Sharpe ratio is the maximum of 400 noisy estimates. Bailey and López de Prado give two corrections.
The **DSR** asks whether the selected Sharpe ratio beats the maximum you would expect from that many
unskilled trials, allowing for skewness, fat tails and sample length. The **PBO** reuses the data: it
splits the history into blocks, picks the best rule on every half of the blocks, and checks how that
rule ranks on the other half. Both are in `lilavati.core.backtest_overfitting`
([validation](../validation/financial_ml.md#backtest-statistics-corebacktest_overfittingpy)).

## 2. A rule tried 400 ways

The experiment follows the seasonal example in Bailey, Borwein, López de Prado and Zhu (2017),
section 6: enter on a given day of each month, hold for some days, long or short. Twenty entry days,
ten holding periods and two sides make 400 rules. The market is a random walk of 1,120 daily returns
(56 months of 20 days), optionally with a real effect: the first five days of every month drift up.

```python title="overfit_lab.py"
"""A seasonal trading rule tried over many settings, as in Bailey et al. (2017), section 6."""

import itertools

import numpy as np
from scipy.stats import kurtosis, skew

from lilavati.core.backtest_overfitting import deflated_sharpe_ratio, probability_of_backtest_overfitting

DAYS_PER_MONTH, MONTHS = 20, 56  # 1,120 trading days: a multiple of 16 for CSCV


def market(seed: int, effect: float = 0.0) -> np.ndarray:
    """Daily returns (decimals) of a random walk; ``effect`` shifts the first 5 days of every month by
    that many daily standard deviations (a real monthly seasonal)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=DAYS_PER_MONTH * MONTHS)
    r[(np.arange(r.size) % DAYS_PER_MONTH) < 5] += effect * 0.01
    return r


def strategies(returns: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """One column per rule: enter on day `entry` of each month, hold `hold` days, long (+1) or short (-1)."""
    day = np.arange(returns.size) % DAYS_PER_MONTH
    grid = list(itertools.product(range(DAYS_PER_MONTH), range(1, 11), (1, -1)))
    cols = [side * returns * ((day >= entry) & (day < entry + hold)) for entry, hold, side in grid]
    return np.column_stack(cols), grid


def evaluate(m: np.ndarray, grid: list, n_trials: float | None = None) -> dict:
    """The best rule in-sample, its annualised Sharpe ratio, its DSR and the search's PBO."""
    sr = m.mean(axis=0) / m.std(axis=0, ddof=1)  # daily Sharpe ratio of every rule
    best = int(np.argmax(sr))
    x = m[:, best]
    dsr = deflated_sharpe_ratio(sharpe_ratio=sr[best], n_obs=len(x), n_trials=n_trials or len(grid),
                                sharpe_ratio_variance=sr.var(ddof=1), skewness=skew(x),
                                kurtosis=kurtosis(x, fisher=False))
    pbo = probability_of_backtest_overfitting(m, n_partitions=16)
    return {"rule": grid[best], "sharpe_annual": sr[best] * np.sqrt(252), "dsr": dsr.deflated_sharpe_ratio,
            "pbo": pbo.pbo, "oos_loss": pbo.prob_oos_loss}
```

Units: the Sharpe ratios passed to `deflated_sharpe_ratio` are daily (per observation), as is the
variance of the 400 Sharpe ratios; kurtosis is raw (3 for a normal distribution), hence
`fisher=False`.

## 3. No edge at all

```python
from overfit_lab import evaluate, market, strategies

m, grid = strategies(market(seed=7))  # a random walk: no rule has an edge
r = evaluate(m, grid)
print(f"{len(grid)} rules; best {r['rule']} (entry day, holding days, side)")
print(f"annualised Sharpe {r['sharpe_annual']:.2f}, DSR {r['dsr']:.3f}, PBO {r['pbo']:.3f}")
```

```text
400 rules; best (19, 1, -1) (entry day, holding days, side)
annualised Sharpe 1.43, DSR 0.010, PBO 0.125
```

Tested as if it were the only rule tried, a Sharpe ratio of 1.43 on four and a half years of data
looks real: its Probabilistic Sharpe Ratio against zero (`probabilistic_sharpe_ratio`) is 0.9997.
The DSR, which allows for the other 399 rules, gives a 1% probability that its true Sharpe ratio
exceeds the maximum expected from 400 unskilled trials. This single PBO is low (0.125); section 5
shows how much one PBO estimate varies.

## 4. A real effect

```python
from overfit_lab import evaluate, market, strategies

m, grid = strategies(market(seed=7, effect=0.25))  # days 0-4 of every month drift up by 0.25 sd
r = evaluate(m, grid)
print(f"best {r['rule']}: annualised Sharpe {r['sharpe_annual']:.2f}, DSR {r['dsr']:.3f}, PBO {r['pbo']:.3f}")
```

```text
best (0, 5, 1): annualised Sharpe 2.06, DSR 0.048, PBO 0.026
```

The search finds the planted rule (enter on day 0, hold 5 days, long). PBO is low, as it should be.
The DSR is also low, which is wrong: section 6 explains why.

## 5. Twenty markets of each kind

One dataset gives one estimate. Repeat with twenty seeds (about a minute):

```python
import numpy as np

from overfit_lab import evaluate, market, strategies

for effect in (0.0, 0.25):
    runs = [evaluate(*strategies(market(seed, effect))) for seed in range(20)]
    found = sum(r["rule"] == (0, 5, 1) for r in runs)
    print(f"effect {effect}: Sharpe {np.mean([r['sharpe_annual'] for r in runs]):.2f}, "
          f"DSR {np.mean([r['dsr'] for r in runs]):.3f}, PBO {np.mean([r['pbo'] for r in runs]):.3f} "
          f"(range {min(r['pbo'] for r in runs):.3f} to {max(r['pbo'] for r in runs):.3f}), "
          f"true rule found {found}/20")
```

```text
effect 0.0: Sharpe 1.14, DSR 0.253, PBO 0.501 (range 0.125 to 0.877), true rule found 0/20
effect 0.25: Sharpe 1.99, DSR 0.367, PBO 0.096 (range 0.000 to 0.311), true rule found 14/20
```

Averaged over markets, PBO is one half when nothing is real (the in-sample winner's out-of-sample rank
is uniform) and near zero when the effect is real. A single estimate ranges from 0.125 to 0.877 on
noise, because its 12,870 combinations all reuse one history.

## 6. The DSR needs the number of independent trials

```python
from overfit_lab import evaluate, market, strategies

m, grid = strategies(market(seed=7, effect=0.25))
for n in (5, 20, 50, 200, 400):
    print(f"N = {n:3}: DSR {evaluate(m, grid, n_trials=n)['dsr']:.3f}")
```

```text
N =   5: DSR 0.984
N =  20: DSR 0.739
N =  50: DSR 0.436
N = 200: DSR 0.115
N = 400: DSR 0.048
```

The DSR's threshold is the expected maximum of N *independent* unskilled trials. These 400 rules are
not independent: every long rule has a short mirror image, and rules with neighbouring entry days and
holding periods hold mostly the same days. Counting them as 400 makes the threshold too high, and the
real rule fails. Counting too few lets noise through. The DSR is only as good as your estimate of the
effective number of trials; PBO does not need one.

## 7. Purged and embargoed cross-validation

Triple-barrier labels (`FinancialMLEngine.apply_triple_barriers`, López de Prado 2018, ch. 3) span
several bars: a label decided at bar 15 used prices up to
bar 15. A fold that trains on bar 12's label and tests on bar 14's leaks the test prices into
training. Purging drops every training label whose interval meets a test label; the embargo drops a
few more bars after each test fold, because returns are serially correlated.

```python
import numpy as np

from lilavati.core.backtest_overfitting import combinatorial_purged_splits, purged_kfold_splits
from lilavati.core.financial_ml import FinancialMLEngine
from overfit_lab import market

prices = 100 * np.exp(np.cumsum(market(seed=7)))
labels = FinancialMLEngine.apply_triple_barriers(prices, holding_horizon=10, take_profit_pct=0.02,
                                                 stop_loss_pct=0.02)
start = [lab.index for lab in labels]  # each label runs from its entry bar ...
end = [lab.touch_index for lab in labels]  # ... to the bar that touched a barrier
for i, s in enumerate(purged_kfold_splits(start, end, n_splits=5, embargo=10)):
    print(f"fold {i}: test {len(s.test)}, purged {len(s.purged)}, embargoed {len(s.embargoed)}, train {len(s.train)}")
c = combinatorial_purged_splits(start, end, n_groups=6, n_test_groups=2, embargo=10)
print(f"{len(c.splits)} combinatorial splits give {len(c.paths)} backtest paths")
```

```text
fold 0: test 222, purged 6, embargoed 10, train 872
fold 1: test 222, purged 13, embargoed 10, train 865
fold 2: test 222, purged 16, embargoed 10, train 862
fold 3: test 222, purged 11, embargoed 10, train 867
fold 4: test 222, purged 8, embargoed 0, train 880
```

```text
15 combinatorial splits give 5 backtest paths
```

The last fold has no embargo: nothing follows it. Combinatorial purged cross-validation (6 groups, 2
tested at a time) gives 15 train/test splits whose test predictions combine into 5 complete backtest
paths, so a strategy is judged on a distribution of out-of-sample paths rather than one. The splits
are tested against an independent implementation of the definitions; they are not yet checked
against the book's code or a reference library ([validation](../validation/financial_ml.md)).

## 8. From an agent

The DSR and PBO are MCP tools on `lilavati-mcp-financial-ml` ([MCP setup](../mcp-setup.md)). The
call below reproduces the published worked example (López de Prado 2015, slides 13 and 14, from
Bailey and López de Prado 2014): a daily strategy with annualised Sharpe 2.5 after 100 trials, five
years of data, skewness −3 and kurtosis 10:

```python
import asyncio
import json

from fastmcp import Client

from lilavati.mcp_servers.financial_ml_server import mcp


async def ask() -> dict:
    async with Client(mcp) as client:  # in process; an agent connects to lilavati-mcp-financial-ml
        result = await client.call_tool("compute_deflated_sharpe_ratio", {
            "sharpe_ratio": 2.5, "n_obs": 1250, "n_trials": 100, "sharpe_ratio_variance": 0.5,
            "skewness": -3, "kurtosis": 10, "periods_per_year": 250})
        return json.loads(result.content[0].text)


d = asyncio.run(ask())
print(f"DSR {d['deflated_sharpe_ratio']:.4f}, SR0 {d['expected_max_sharpe_ratio']:.4f} (daily), "
      f"status: {d['validation']['status']}")
```

```text
DSR 0.9004, SR0 0.1132 (daily), status: reference-checked
```

## What can go wrong

- **Counting trials.** The DSR needs the number of *independent* trials (section 6). Correlated trials
  count for fewer; nothing in the data tells you how many, so state the number you used and why.
- **Units.** Sharpe ratios and their variance must be per observation, or pass `periods_per_year`.
  Kurtosis is raw: `scipy.stats.kurtosis` returns excess kurtosis unless `fisher=False`, and passing 0
  for a normal distribution raises an error (raw kurtosis is at least 1 + skewness²).
- **Rows not a multiple of the blocks.** CSCV needs equal blocks: 1,120 rows suit 16 blocks; drop rows
  yourself rather than let the code do it silently.
- **A trial with no variance.** A rule that never trades in one half has no Sharpe ratio there, and the
  code raises. (While building this, a variance computed from sums of squares came out as 1e-19 instead
  of 0 and slipped past the check; it is now computed from centred moments.)
- **Labels that span a fold.** With long triple-barrier horizons, purging can leave a fold with no
  training data; the code raises instead of training on nothing.

## What this shows, and what it does not

- It shows that the best of many backtests can look significant on pure noise, and that both
  statistics flag it; and that PBO separates a planted effect from noise on average.
- It does not show that a low PBO proves a strategy works: PBO measures whether selection generalises
  across halves of one history, not whether the future resembles it.
- The DSR depends on the number of independent trials, which you have to estimate; a wrong count
  makes it too strict or too lenient (section 6).
- A single PBO estimate is noisy (0.125 to 0.877 on noise here); report it with the experiment
  design, and repeat where you can.

## Exercises

1. Count the trials honestly: drop the short rules (side −1) and recompute the DSR with N = 200. How
   does it compare with N = 400 in section 6?
2. Set `effect=0.1`. How often is the planted rule found in twenty seeds, and what happens to PBO?
3. Change `n_partitions` to 8 in `evaluate`. How many combinations are there, and does PBO change?
4. In section 7, raise `holding_horizon` to 30. Why does the number of purged labels rise?

## Using this with Claude Code and MCP

- **From an AI assistant:** the `lilavati-mcp-financial-ml` server's `compute_deflated_sharpe_ratio`
  and `estimate_probability_of_backtest_overfitting` tools take the statistics or the matrix of
  returns (one column per configuration tried) and return the DSR or the PBO, with each engine's
  validation status ([MCP setup](../mcp-setup.md)). Ask the assistant how many independent trials it
  assumed.
- **Contributing:** the [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
  holds the rules learned here (centred moments for variances; a statistic that needs a count of
  independent trials says so).

## For instructors

- **Discussion:** the DSR rejected a real effect in section 6. Is that a flaw of the DSR or of how the
  trials were counted? What would you report to an investment committee?
- **Lab:** each group designs its own family of rules (moving averages, breakouts, calendar rules), runs
  sections 3 to 5 on noise and on a planted effect, and reports PBO and DSR with the trial count it
  defends.
- **Assessment idea:** give students one impressive backtest and the list of everything that was tried;
  ask for the DSR, the PBO and a one-paragraph verdict.

## References

- Bailey, D. H. and López de Prado, M. (2014), "The deflated Sharpe ratio: correcting for selection
  bias, backtest overfitting and non-normality", *Journal of Portfolio Management* 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017), "The probability of backtest
  overfitting", *Journal of Computational Finance* 20(4).
- López de Prado, M. and Bailey, D. H. (2021), "The false strategy theorem", *American Mathematical
  Monthly* 128(9).
- López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, ch. 3, 7 and 12.
- López de Prado, M. (2015), "Advances in quantitative meta-strategies", conference slides.
