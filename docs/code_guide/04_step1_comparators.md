## 4. Step 1 and the comparators: causal discovery on the transition table

### 4.0 Overview

**What this chapter covers.** Step 1 of the protocol (§7.2) learns **one causal map from all participants pooled**, on a table of year-to-year transitions. It has two jobs:

1. **The reported map**, with marks that flag possible hidden confounding (`reported_map`).
2. **The support $\mathcal S$ of Steps 2 to 5**: the drift entries the engine may use. It is the expert map's links, plus every lagged adjacency the fast configuration finds, plus every own rate (`select_support`). The same rule is re-applied in every bootstrap replicate of H-P (`make_support_fn`), so the test is calibrated after the selection (§0.5, idea 4).

The same learner, run per phenotype on equal-size subsamples, gives **the separate-graphs reference** of §7.3.5 (`separate.py`). The **named comparators** of §8.4 (`comparators/`) are a descriptive benchmark: they show what established methods would report as a "mechanism difference" when the phenotypes differ only in speed. Four are coded here: C3, C5, C11 and C12.

| file | lines | what it holds | protocol | guide |
|---|---|---|---|---|
| [`step1.py`](../../src/pdx/step1.py) | 282 | the two-slice table with tiers; Fast Causal Inference (FCI) with tiered knowledge; two conditional-independence (CI) tests with test-wise deletion; the replacement of one causal-learn function; lagged adjacencies; the support; the reported map; sortability | §6.2, §6.3, §7.2 | §4.1–§4.16 |
| [`separate.py`](../../src/pdx/separate.py) | 75 | separate graphs on equal subsamples, permutation reference, family-wise threshold | §7.3.5 | §4.17–§4.21 |
| [`comparators/__init__.py`](../../src/pdx/comparators/__init__.py) | 31 | the status of C1–C12 | §8.4 | §4.22 |
| [`comparators/run.py`](../../src/pdx/comparators/run.py) | 161 | wrappers for C3, C5, C11 and C12 on a `Panel` | §8.4 | §4.23–§4.29 |
| [`comparators/dynotears.py`](../../src/pdx/comparators/dynotears.py) | 104 | DYNOTEARS (Pamfil 2020), re-implemented | §8.4 (C12) | §4.30–§4.36 |
| [`comparators/ts_ldiffpc.py`](../../src/pdx/comparators/ts_ldiffpc.py) | 273 | tsLDiffPC and tsDCI (Ruer 2026), re-implemented | §8.4 (C5) | §4.37–§4.47 |

The subsections follow the source, file by file, in the order of this table; §4.N collects the notes for the author. The wrappers in `run.py` come before the algorithms they call, so §4.26 and §4.27 point forward to §4.30–§4.47.

**The data flow.**

```text
Panel: panel.y (scores), panel.t, panel.year (grid years), panel.z (era, age at onset, sex, duration), panel.dose
  │  data.two_slice_table: one row per pair of consecutive grid years (t, t+1) of one participant
  ▼
step1_table → X = [ z | (interval) | Y_t | (dose_t, dose_t+1) | Y_t+1 ],  tiers,  who (row → participant)
  │  fci_graph: causal-learn's FCI with tiers and the exogenous rule; CI test pdx_fisherz or
  │             pdx_cluster_fisherz (test-wise deletion); fci_orient_bk swapped for _orient_tiers
  ▼
endpoint matrix G  (G[a, b] = mark at column a on the edge a – b)
  ├─ lagged_adjacency → L[to, from] ─ select_support: L ∪ expert ∪ diagonal ─→ support S of the engine (§2, §3, §5)
  ├─ reported_map → edges with marks, row and participant counts (aggregates)
  └─ per-phenotype subsamples → lagged_adjacency, lagged_directed → separate.py: D(e), threshold, candidates

comparators/run.py: _pairs (complete consecutive visits ≤ 1.5 years apart), pooled standardisation
  ├─ C12: DYNOTEARS per phenotype, discordant links            (dynotears.py)
  ├─ C5 : tsDCI / tsLDiffPC difference graph, root causes       (ts_ldiffpc.py)
  ├─ C3 : permutation test of yearly least-squares networks
  └─ C11: separate graphs read naively; v0.3 interaction tests
```

**New notation for this chapter** (the shared notation of §0.4 still holds).

| symbol | meaning | code |
|---|---|---|
| $r$; $i(r)$ | a row of the two-slice table (one transition from grid year $t$ to $t+1$); its participant | a row of `X`; `who[r]` |
| $\tau(v)$ | the tier of column $v$ (0 = era, 1 = other baseline, 2 = year $t$, then later) | `tiers` |
| $U, V$; $\mathbf S$, $k=\lvert\mathbf S\rvert$ | the two columns a CI test compares ($U$ has the lower column index); the conditioning set | causal-learn's `X`, `Y`, `condition_set` |
| $R$, $n$ | the rows with $U$, $V$ and all of $\mathbf S$ observed (test-wise deletion); their number | `rows`, `n` |
| $N_c$ | the number of participants among those rows (the clusters of one test) | `ncl` |
| $e_U, e_V$ | residuals of $U$ and $V$ after least squares on $[1,\mathbf S]$ over $R$ | `ex`, `ey` |
| $\hat\rho$ | sample partial correlation of $U$ and $V$ given $\mathbf S$ | `r` |
| $\alpha$ | level of every CI test inside FCI (0.05) | `cfg.alpha` |
| $\mathcal G[a,b]$ | the endpoint matrix: mark at column $a$ on the edge $a$–$b$; −1 tail, 1 arrowhead, 2 circle, 0 no edge | `G` |
| $L_{jk}$ | 1 if domain $k$ in year $t$ and domain $j$ in year $t+1$ are adjacent; stored **[to, from]**, like the support and $A_{jk}$ | `lagged_adjacency(...)[j, k]` |
| $\hat p_g(e)$; $D(e)$ | selection frequency of link $e$ over the subsamples of phenotype $g$; $D(e)=\hat p_0(e)-\hat p_1(e)$ | `lagged_frequencies`; `D_adjacency`, `D_directed` |
| $W$, $A$ (C12 only) | DYNOTEARS same-time and lagged weights, stored **[from, to]** ($W_{ab}$: $a\to b$) | `W`, `A` in `dynotears.py` |
| $Z^{(1)}, Z^{(2)}$ (C5 only) | the two regimes' windows: columns $0..d-1$ = lag block $x(t-1)$, $d..2d-1$ = current block $x(t)$ | `Z1`, `Z2` |

In this chapter $A$ means DYNOTEARS' lagged weights only inside §4.26 and §4.30–§4.36; elsewhere it is the engine's drift.

**Two matrix conventions.** The support, `lagged_adjacency`, `lagged_directed`, the separate-graphs statistics and every comparator output are **[to, from]**: row $j$, column $k$ is the link $k\to j$, exactly like the drift entry $A_{jk}$. DYNOTEARS' $W$ and $A$, and the `directed` argument of `sortability`, use the usual graph convention **[from, to]**. causal-learn's endpoint matrix is neither: $\mathcal G[a,b]$ is the *mark at $a$*. Every place where the code converts between them is pointed out below.

**How the examples were run.** Every script is in [`docs/guide_examples/s04/`](../guide_examples/s04/) and uses synthetic data only: hand-made linear Gaussian data with a known graph, or `pdx.sim.scenarios.generate`. None reads `data/` or writes `results/`. Run one from the repository folder:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s04/ex02_fisherz.py
```

The environment: Python 3.13.12, numpy 2.5.4, scipy 1.18.1, causal-learn 0.1.4.8, statsmodels 0.15.0. Each script runs in under 10 seconds; printed timings vary from run to run. causal-learn's FCI can depend on Python's string-hash seed (Notes 4.N, item 2); the outputs pasted below were identical under hash seeds 0 to 3. The unit tests of this chapter (`tests/test_step1.py`, five tests; `tests/test_comparators.py`, one test) pass in about 2 seconds each.

---

### 4.1 The module docstring, `MARK`, `_REGISTERED`, `_TESTS` — what Step 1 promises

[`step1.py:1`](../../src/pdx/step1.py#L1), [`step1.py:35`](../../src/pdx/step1.py#L35)

**What it does.** The docstring (lines 1–23) states the design in four points:
- **The learner** is FCI with tiered background knowledge from causal-learn, run on the pooled two-slice table. The phenotype labels are *not* in the graph (protocol §1.3: the phenotype is computed from symptom scores, so it cannot be an exogenous context node).
- **The tiers**: era | other baseline variables | year $t$ | treatment at $t+1$ | the rest of year $t+1$. Nothing but era causes age at onset or sex.
- **The tests**: `"fisherz"`, the uncorrected Fisher-z test on the normal scores, used in every resample; and `"cluster_fisherz"`, a partial-correlation test with a participant-clustered sandwich variance. Both use test-wise deletion. Any other causal-learn test name runs on complete rows.
- **The support rule**: a drift entry $X\to Y$ enters when the map has an adjacency from $X$ in year $t$ to $Y$ in year $t+1$, with any marks, or when the expert map proposes it. Same-year adjacencies add no entry.

Three module constants follow:
- `MARK = {-1: "-", 1: ">", 2: "o"}` turns causal-learn's endpoint codes into characters for `reported_map`.
- `_REGISTERED` remembers whether the two custom tests are already registered with causal-learn (§4.3).
- `_TESTS = {"fisherz": "pdx_fisherz", "cluster_fisherz": "pdx_cluster_fisherz"}` maps the configuration's short names to the registered names. It is the reason `Step1Config(test="fisherz")` runs **this package's** Fisher-z test, with test-wise deletion, and not causal-learn's own `fisherz`.

**Inputs → outputs.** Constants only.

**The mathematics.** None here; the tiers and tests are derived in §4.5–§4.8.

**Worked example.** `MARK` is used in every example that prints an edge, for instance §4.7: an endpoint pair (circle at $a$, arrowhead at $b$) prints as `a o-> b`.

**Design choices and pitfalls.**
- The short name `"fisherz"` silently means `pdx_fisherz`. Someone reading a configuration file might expect causal-learn's test. The two agree exactly on complete data (§4.5, example), so the difference matters only when values are missing.
- The docstring's tier list and the exogenous rule differ slightly from protocol §6.2 (see §4.8 and Notes 4.N).

### 4.2 `Step1Config` — the knobs of one Step 1 fit

[`step1.py:41`](../../src/pdx/step1.py#L41)

**What it does.** A dataclass with the settings of one FCI fit:

| field | default | meaning |
|---|---|---|
| `alpha` | 0.05 | level of every CI test inside FCI |
| `depth` | 3 | the largest conditioning set in FCI's adjacency search (−1: no cap). The comment calls it a "separating-set cap"; §4.8 shows it caps only the first phase |
| `test` | `"fisherz"` | `"fisherz"` (fast) or `"cluster_fisherz"`; any other causal-learn name runs on complete rows |
| `era_col` | 0 | the column of $z$ that holds era; it gets its own first tier |
| `exogenous_cols` | (1, 2) | the columns of $z$ that nothing but era may cause: age at onset and sex |
| `add_interval` | False | the interval rule of §6.3: partial the interval out of every column |
| `include_dose` | False | add dose at $t$ (year-$t$ tier) and dose at $t+1$ (its own tier) |
| `max_dt` | 1.5 | the largest interval in years for a "consecutive" pair, used only when the panel has no grid years |
| `max_path_length` | −1 | passed to FCI; it limits only the discriminating-path rule R4 (§4.8) |

**Inputs → outputs.** Keyword arguments → a configuration object. The functions of this module take it as `cfg`.

**The mathematics.** None.

**Worked example** ([`ex01_config.py`](../guide_examples/s04/ex01_config.py)). The three configurations the code actually uses:

```python
print("default (= fast):     ", Step1Config())
s1 = tomllib.load(f)["step1"]                     # configs/protocol_v0.5.toml
print("run_hp_real.py:       ", Step1Config(alpha=s1["alpha"], depth=s1["depth_fast"], test=s1["test_fast"]))
print("reported_map default: ", Step1Config(test="cluster_fisherz", depth=4))
```

```text
default (= fast):      Step1Config(alpha=0.05, depth=3, test='fisherz', era_col=0, exogenous_cols=(1, 2), add_interval=False, include_dose=False, max_dt=1.5, max_path_length=-1)
run_hp_real.py:        Step1Config(alpha=0.05, depth=3, test='fisherz', era_col=0, exogenous_cols=(1, 2), add_interval=False, include_dose=False, max_dt=1.5, max_path_length=-1)
reported_map default:  Step1Config(alpha=0.05, depth=4, test='cluster_fisherz', era_col=0, exogenous_cols=(1, 2), add_interval=False, include_dose=False, max_dt=1.5, max_path_length=-1)
type of alpha read from the TOML file: float
```

The real-data script builds the default (the `[step1]` table of the TOML file has `alpha = 0.05`, `depth_fast = 3`, `test_fast = "fisherz"`), plus two command-line flags that switch on the interval rule and the dose tiers. The reported map uses the clustered test with a cap of 4, marked provisional in its docstring (WP3 chooses at most three configurations per learner).

**Design choices and pitfalls.**
- `era_col` and `exogenous_cols` index the columns of `panel.z`. They assume the order era, age at onset, sex, duration, which is the order `ppmi.analysis_panel` builds (`ppmi.py:423`). A panel with another column order would put the exogenous rule on the wrong covariates, silently.
- causal-learn checks types strictly: its adjacency search raises `TypeError` unless `alpha` is a Python `float` and `depth` a Python `int`. A `numpy.float64` alpha fails. TOML values are plain floats, so the scripts are safe; the last line of the example confirms it.

### 4.3 `_register_tests` — two CI tests inside causal-learn

[`step1.py:53`](../../src/pdx/step1.py#L53)

**What it does.** Defines two CI-test classes and registers them with causal-learn under the names `"pdx_fisherz"` and `"pdx_cluster_fisherz"`. A module flag makes the registration happen once per process.

**Inputs → outputs.** None → a side effect: two entries in causal-learn's registry `causallearn.utils.cit._custom_ci_tests`.

**How the registration works.** When FCI is called as `fci(table, "pdx_fisherz", ..., clusters=who)`, causal-learn runs `CIT(table, method="pdx_fisherz", clusters=who)`. `CIT` finds the name in its registry and builds the class with the data and every extra keyword argument (`causallearn/utils/cit.py:62–63`). That is how the row-to-participant index `who` reaches the test. The class must inherit from `CIT_Base`, which supplies the p-value cache and the formatting of `(X, Y, S)` into a cache key.

The two classes:
- `PartialCorrelation` (`robust = False`, label `"pdx_fisherz"`): the Fisher-z test of §4.5;
- `ClusterPartialCorrelation` (`robust = True`, label `"pdx_cluster_fisherz"`): the same class with the clustered variance of §4.6. It changes only the flag and the label.

**The mathematics.** None here (§4.5, §4.6).

**Worked example.** [`ex02_fisherz.py`](../guide_examples/s04/ex02_fisherz.py) calls `step1._register_tests()` and then `CIT(data, "pdx_fisherz")`, which is exactly what FCI does; see §4.5.

**Design choices and pitfalls.**
- The registry is module-level in causal-learn, so it is per process. The H-P workers are started with `spawn` (`parallel.py`), so each worker registers again on its first `fci_graph` call. That is why the guard is a module flag and not a one-off at import.
- The cache lives inside the test object, and FCI builds one test object per call. Nothing is cached between fits, so a bootstrap replicate cannot reuse p-values from the observed data.

### 4.4 `PartialCorrelation.__init__` — preparing test-wise deletion

[`step1.py:65`](../../src/pdx/step1.py#L65)

**What it does.** Runs once per FCI fit, before any test:
1. `super().__init__(data)`: causal-learn's base class stores the data, its shape and an empty p-value cache.
2. `check_cache_method_consistent(self.label, "test-wise deletion")`: records the method name. It also sets `self.method`, which the key formatter reads later.
3. `assert_input_data_is_valid(allow_nan=True)`: NaN is allowed (missing scores), infinity is not.
4. The cluster index: `clusters` (the participant of each row) or, if none is given, one cluster per row. `np.unique(..., return_inverse=True)` relabels the participants $0,1,\dots$.
5. The observed mask `obs = ~isnan(data)`.
6. A standardised copy: each column minus its mean, divided by its standard deviation, both over the values present; NaN then set to 0 by `nan_to_num`.

**Inputs → outputs.** `data` (rows × columns, NaN allowed) and `clusters` (one label per row, or None) → an object holding `Z`, `obs` and `cid`.

**The mathematics.** Standardising a column changes neither a partial correlation nor the $t$ or $z$ statistic of a regression slope: both are invariant to affine changes of each variable when the regression has an intercept. It only improves the numerical conditioning of the least-squares solves. The zeros written into missing cells are never used, because every test selects its rows through `obs` first (§4.5).

**Worked example.** Part 4 of the example in §4.5 runs this constructor on a table with 10% of each column missing.

**Design choices and pitfalls.**
- The means and SDs are over **all** values present in a column, not over the rows of a particular test. This does not matter, by the invariance above: each test re-centres its residuals.
- A column that is entirely missing gives a NaN mean (numpy warns) and a column of zeros. Every test involving it then has no rows and returns $p=1$ (§4.5).

### 4.5 `PartialCorrelation.__call__`, plain branch — the Fisher-z test with test-wise deletion

[`step1.py:75`](../../src/pdx/step1.py#L75) (the Fisher-z lines are [`step1.py:90–92`](../../src/pdx/step1.py#L90))

**What it does.** Returns the p-value of $U\perp\!\!\!\perp V\mid\mathbf S$ using only the rows where $U$, $V$ and every member of $\mathbf S$ are observed. FCI calls it as `test(X, Y, S)` with column indices.

**Inputs → outputs.** Two column indices and a conditioning set → a p-value in $[0,1]$. Side effect: the p-value is cached under a key such as `"2;3|0.1"` (the two indices sorted, then the sorted set).

**The mathematics.**

*Partial correlation.* For jointly Gaussian variables, $U\perp\!\!\!\perp V\mid\mathbf S$ holds exactly when the partial correlation $\rho_{UV\cdot\mathbf S}$ is zero. Its sample version is the correlation of the two least-squares residuals. With $C=[\mathbf 1,\ \mathbf S]$ the $n\times(k+1)$ design over the test's rows $R$,

$$
e_U=(I-C(C^\top C)^{-1}C^\top)\,U,\qquad e_V=(I-C(C^\top C)^{-1}C^\top)\,V,\qquad
\hat\rho=\frac{e_U^\top e_V}{\sqrt{(e_U^\top e_U)(e_V^\top e_V)}} .
$$

This equals $-P_{UV}/\sqrt{P_{UU}P_{VV}}$, with $P$ the inverse of the correlation matrix of $(U,V,\mathbf S)$, which is how causal-learn's own `fisherz` computes it.

*Fisher's transformation.* Under Gaussianity, $\hat\rho$ from $n$ rows with $k$ conditioning variables is distributed like an ordinary correlation from $n-k$ rows. Fisher's

$$
z=\tfrac12\ln\frac{1+\hat\rho}{1-\hat\rho}=\operatorname{artanh}\hat\rho
$$

is then approximately normal with mean $\operatorname{artanh}\rho$ and variance $1/(n-k-3)$. Under $H_0:\rho_{UV\cdot\mathbf S}=0$,

$$
T=\sqrt{n-k-3}\;\lvert z\rvert\ \approx\ \lvert N(0,1)\rvert,\qquad p=2\{1-\Phi(T)\}.
$$

*Test-wise deletion.* $R=\{r:\ U_r, V_r, S_{1r},\dots,S_{kr}\ \text{observed}\}$, so $n=\lvert R\rvert$ changes from test to test. Each test is the complete-case test of its own variables. It answers the intended question when keeping only the complete rows does not change the conditional independence being tested; a sufficient condition is that the chance of a row being complete does not depend on both $U$ and $V$ given $\mathbf S$ (selecting on a common effect of $U$ and $V$ would create a dependence). For FCI this has to hold for every test, which is why the protocol asks for a missingness graph in WP1 (§7.2).

**From the formulas to the code, line by line.**

| line | code | formula |
|---|---|---|
| 76 | `get_formatted_XYZ_and_cachekey` | sorts so that $U$ has the lower index; builds the cache key |
| 79–80 | `rows = obs[:, [X, Y] + S].all(1)`; `n = rows.sum()` | $R$ and $n$ |
| 81–82 | `p = 1.0`; `if n > len(S) + 3` | the statistic needs $n-k-3>0$; otherwise $p=1$ |
| 83–87 | `C = [1, Z_S]`; `lstsq`; `xy - C @ coef` | $e_U$, $e_V$ (one solve for both columns) |
| 88 | `ex, ey = ... - mean` | re-centring (needed only when $\mathbf S$ is empty and $C$ is absent) |
| 89 | `sxx, syy` | $e_U^\top e_U$, $e_V^\top e_V$ |
| 91 | `r = clip(ex @ ey / sqrt(sxx * syy), ±(1 − 1e−12))` | $\hat\rho$, kept off $\pm1$ so that artanh stays finite |
| 92 | `2 * norm.sf(sqrt(n − len(S) − 3) * abs(arctanh(r)))` | $p=2\{1-\Phi(T)\}$ |

**How the block was formed.** causal-learn's own `fisherz` inverts the correlation matrix of the whole table, which needs complete rows. Test-wise deletion needs the rows to change with $(U,V,\mathbf S)$, so a precomputed matrix cannot be used. The residual form computes $\hat\rho$ from the rows of each test directly, and one `lstsq` call residualises both columns at once.

**Worked example** ([`ex02_fisherz.py`](../guide_examples/s04/ex02_fisherz.py)). Two conditioning variables $S_1,S_2$, then $X$ and $Y$ with a small effect $X\to Y$:

```python
x = 0.7 * s1 + rng.normal(size=n)
y = 0.5 * s1 - 0.4 * s2 + 0.12 * x + rng.normal(size=n)          # n = 400
data = np.column_stack([s1, s2, x, y])                            # columns 0, 1 = S; 2 = X; 3 = Y
# by hand: residualise on [1, S], correlate, Fisher's z ...
cit = CIT(data, "pdx_fisherz"); cit(2, 3, [0, 1])                 # as FCI calls it
CIT(data, "fisherz")(2, 3, [0, 1])                                # causal-learn's own test
miss[rng.random(miss.shape) < 0.10] = np.nan                      # then 10% of each column missing
```

```text
by hand:      r = 0.091120, z = 0.091373, sqrt(n-|S|-3)|z| = 1.8160, p = 0.0693691
pdx_fisherz:  p = 0.0693691   (cache key '2;3|0.1')
fisherz:      p = 0.0693691
S = []      rows used = 321,  pdx p = 1.75788e-10,  by hand on those rows = 1.75788e-10
S = [0]     rows used = 300,  pdx p = 0.184645,  by hand on those rows = 0.184645
S = [0, 1]  rows used = 261,  pdx p = 0.222896,  by hand on those rows = 0.222896
rows complete in all four columns: 261
5 rows, |S| = 2: p = 1.0
```

Reading the output:
- On complete data the three numbers agree to every printed digit: $\sqrt{400-2-3}=19.875$ and $19.875\times0.091373=1.8160$.
- With missing values, each test uses its own rows: 321 for the marginal test, 300 given $S_1$, 261 given both. Complete-case analysis would use 261 rows for every test.
- With too few rows ($n=5\le k+3=5$) the test returns $p=1$. Inside FCI that means "independent": the edge is removed and the set is recorded as separating.

**Design choices and pitfalls.**
- **"Cannot test" becomes "independent".** $p=1$ is returned when $n\le k+3$, and also when a residual has zero variance (line 90's `sxx > 0 and syy > 0` fails). At PPMI's size neither should happen with whole domains, but it can in small subsamples (separate graphs) or with a nearly constant column. Returning NaN and keeping the edge would be the conservative choice (Notes 4.N).
- The uncorrected test treats rows as independent. Rows of one participant are not, so its p-values are too small when the residuals are correlated within participant (§4.6 measures this). The protocol accepts this for the fast configuration, because every resampling decision is calibrated on simulated data with the same participant structure (§7.2).

### 4.6 `PartialCorrelation.__call__`, robust branch, and `ClusterPartialCorrelation` — the participant-clustered test

[`step1.py:93`](../../src/pdx/step1.py#L93), [`step1.py:103`](../../src/pdx/step1.py#L103)

**What it does.** With `robust = True`, the same residuals $e_U, e_V$ feed a Wald test of the regression slope of $V$ on $U$ (given $\mathbf S$). The variance is a sandwich that sums the score contributions **within participant**, so it allows any correlation between one participant's transitions and any unequal residual variance.

**Inputs → outputs.** As in §4.5. The clusters are the `clusters` argument of the constructor: in Step 1, `who`, the participant of each row.

**The mathematics.**

*The slope.* In the regression $V=a+\beta U+\gamma^\top\mathbf S+u$, the Frisch–Waugh–Lovell theorem gives

$$
\hat\beta=\frac{e_U^\top e_V}{e_U^\top e_U},\qquad \hat u=e_V-\hat\beta\,e_U ,
$$

where $\hat u$ is exactly the residual of the full regression. $\hat\beta=0$ exactly when $\hat\rho=0$, so the null hypothesis is the same as in §4.5; only the variance differs.

*The clustered (sandwich) variance.* Write $\hat\beta-\beta=(e_U^\top e_U)^{-1}\sum_r e_{U,r}u_r$. Group the sum by participant $c$: $s_c=\sum_{r\in c}e_{U,r}\hat u_r$. Participants are independent, so

$$
\widehat{\operatorname{Var}}(\hat\beta)=\frac{N_c}{N_c-1}\cdot\frac{\sum_{c=1}^{N_c}s_c^2}{\left(e_U^\top e_U\right)^2},\qquad
z=\frac{\hat\beta}{\sqrt{\widehat{\operatorname{Var}}(\hat\beta)}},\qquad p=2\{1-\Phi(\lvert z\rvert)\}.
$$

$N_c$ counts the participants with at least one row in $R$. By the same Frisch–Waugh–Lovell argument, this is exactly the $(U,U)$ element of the full sandwich $(\mathbf H^\top\mathbf H)^{-1}\big(\sum_c \mathbf H_c^\top\hat u_c\hat u_c^\top\mathbf H_c\big)(\mathbf H^\top\mathbf H)^{-1}$ with the design $\mathbf H=[U,\mathbf 1,\mathbf S]$: the row of $(\mathbf H^\top\mathbf H)^{-1}\mathbf H^\top$ that belongs to $U$ is $e_U^\top/(e_U^\top e_U)$.

What it allows, and why it is needed:
- **Within-participant correlation.** A participant contributes one row per pair of consecutive yearly visits. A participant-level speed or load moves all of them together. The naive variance $\hat\sigma^2/e_U^\top e_U$ then understates the true variance by roughly the design effect $1+(m-1)\rho_U\rho_V$, with $m$ rows per participant and $\rho_U,\rho_V$ the within-participant correlations of the two residuals.
- **Unequal variance.** A residual variance that changes with the state (protocol §7.2: "patient-level speed makes residual variance depend on the current state") also breaks the naive variance. The sandwich uses the squared scores themselves.

The small-sample factor is $N_c/(N_c-1)$ only. The usual "CR1" of Stata and statsmodels also multiplies by $(n-1)/(n-p)$, with $p$ the number of regressors; the reference distribution is the normal, not $t_{N_c-1}$. Both differences vanish with hundreds of participants.

**From the formulas to the code.**

| line | code | formula |
|---|---|---|
| 94–95 | `np.unique(self.cid[rows], return_inverse=True)`; `ncl` | relabel the participants present in $R$; $N_c$ |
| 96 | `beta = ex @ ey / sxx` | $\hat\beta$ |
| 97 | `np.bincount(cid, weights=ex * (ey - beta * ex))` | $s_c=\sum_{r\in c}e_{U,r}\hat u_r$ |
| 98 | `score @ score / sxx**2 * ncl / max(ncl - 1, 1)` | $\widehat{\operatorname{Var}}(\hat\beta)$ |
| 99 | `2 * norm.sf(abs(beta) / sqrt(V)) if V > 0 else 1.0` | $p$ |

**Worked example** ([`ex03_cluster_test.py`](../guide_examples/s04/ex03_cluster_test.py)). Sixty participants with five rows each. $X$ and $Y$ are independent given $S$, but each has a participant effect with intra-class correlation 0.8:

```python
x = 0.5 * s + sqrt(0.8) * a[who] + sqrt(0.2) * noise      # a, b: independent participant effects
y = 0.5 * s + sqrt(0.8) * b[who] + sqrt(0.2) * noise
p_code = CIT(data, "pdx_cluster_fisherz", clusters=who)(1, 2, [0])
score = np.bincount(who, weights=ex * (ey - beta * ex)); V = score @ score / (ex @ ex)**2 * G / (G - 1)
ols = sm.OLS(Z[:, 2], [1, Z[:, 1], Z[:, 0]]).fit(cov_type="cluster", cov_kwds={"groups": who})
```

```text
code:        p = 0.395267
by hand:     beta = -0.084621, se = 0.099542, z = -0.8501, p = 0.395267
statsmodels: beta = -0.084621, se = 0.099877 (CR1 with (N-1)/(N-K)); se x sqrt((N-K)/(N-1)) = 0.099542
plain Fisher-z on the same data: p = 0.147688
robust test with X and Y swapped (regression of X on Y): p = 0.369183

rejection rate under the null (400 datasets each, 7.1 s):
design                        pdx_fisherz  pdx_cluster_fisherz
independent rows                    0.048                0.055
participant clusters                0.300                0.040
shared heteroscedasticity 0.4        0.163                0.062
shared heteroscedasticity 0.8        0.450                0.098
```

Reading the output:
- The code, the hand computation and statsmodels agree once statsmodels' extra factor $(N-1)/(N-K)$ is removed ($N=300$ rows, $K=3$ regressors).
- **Size.** With independent rows both tests hold 5% (Monte Carlo SE about 0.011). With participant clusters the plain test rejects 30% of true nulls; the clustered test holds. The design effect predicts the 30%: $1+(5-1)\times0.8^2=3.56$, so the plain standard error is too small by $\sqrt{3.56}=1.89$, and $P(\lvert N(0,1)\rvert>1.96/1.89)=0.30$. When both residual SDs grow with $S$ (SD $e^{0.4S}$ or $e^{0.8S}$), the plain test fails badly; the sandwich is much better but still above 5% in the extreme design, the known small-sample bias of sandwich variances under heavy-tailed scores.
- **Asymmetry.** Swapping $X$ and $Y$ changes the robust p-value (0.395 against 0.369). The plain Fisher-z is symmetric in $U$ and $V$; a sandwich test of a slope is not.

**Design choices and pitfalls.**
- **Which slope is tested is decided by column order.** causal-learn's key formatter always passes the lower column index as `X`, so the code always regresses the later column on the earlier one. In the two-slice table, year-$t$ columns come before year-$t+1$ columns, so a lagged pair is tested as "year $t+1$ on year $t$", the natural direction. A same-year pair is tested in column order, which is arbitrary.
- With `clusters=None` every row is its own cluster, and the test becomes the heteroscedasticity-robust (White, HC0 × $n/(n-1)$) test.
- In the fast configuration (`"fisherz"`) the `clusters` argument is passed but not used.

### 4.7 `_orient_tiers` — tiers give an arrowhead at the later node, nothing more

[`step1.py:112`](../../src/pdx/step1.py#L112)

**What it does.** It replaces causal-learn's `fci_orient_bk` while `fci_graph` runs (§4.8). For every edge, if background knowledge forbids one direction, it puts an **arrowhead** at the node that cannot be the cause, and leaves the mark at the other node as it was (a circle at that point of FCI). causal-learn's own function instead makes the edge fully directed, with a **tail** at the earlier node.

**Inputs → outputs.** `bk` (causal-learn `BackgroundKnowledge`, or None) and `graph` (the partially oriented graph) → the graph modified in place.

**The mathematics.**

*Marks in a partial ancestral graph (PAG).* FCI's output summarises every causal structure, hidden common causes allowed, that implies the same conditional independences. Each edge has a mark at each end:
- an **arrowhead** at $b$ on $a{*\!\!\to}b$ means *$b$ is not an ancestor of $a$* (it does not cause $a$, directly or indirectly);
- a **tail** at $a$ on $a{-\!\!*}b$ means *$a$ is an ancestor of $b$*;
- a **circle** means the mark differs between compatible structures.

*What a tier says.* Tiered knowledge says that a node in a later tier cannot be an ancestor of a node in an earlier tier: year-$t+1$ scores cannot cause year-$t$ scores, and nothing in the table causes era. In a two-slice table, take $a$ in year $t$ and $b$ in year $t+1$:

$$
\tau(b)>\tau(a)\ \Longrightarrow\ b\notin\operatorname{an}(a)\ \Longleftrightarrow\ \text{arrowhead at } b:\quad a\ {*\!\!\to}\ b .
$$

It says nothing about whether $a$ is an ancestor of $b$. An adjacency between $a$ and $b$ can come from a direct effect $a\to b$, or from a hidden common cause $a\leftarrow L\to b$ (for example the participant's own speed, load or overall burden), or both. So the mark at $a$ must be left to the data: a circle, or a tail or an arrowhead if the data imply one.

*Why the tail is wrong.* A tail at $a$ asserts "$a$ causes $b$". Under hidden confounding that is false, and the tail also feeds FCI's later rules (R2, R8–R10 use tails), so the error can spread to other edges. causal-learn 0.1.4.8 imposes that tail on every cross-tier adjacency (`fci_orient_bk`, `causallearn/search/ConstraintBased/FCI.py:269–291`): the yearly map would show `-->` everywhere from year $t$ to year $t+1$, and the protocol's "marks for possible hidden confounding" would never appear.

*What the replacement does.* For an edge with nodes $n_1, n_2$ (in causal-learn's storage order), the pair (mark at $n_1$, mark at $n_2$) becomes

$$
\text{forbidden}(n_2\to n_1):\ \ (\text{unchanged},\ \text{arrowhead}),\qquad
\text{forbidden}(n_1\to n_2):\ \ (\text{arrowhead},\ \text{unchanged}).
$$

| line | code | meaning |
|---|---|---|
| 115–116 | `if bk is None: return` | no knowledge, nothing to do |
| 119 | `for edge in graph.get_graph_edges()` | a fresh list, so editing the graph inside the loop is safe |
| 121–124 | `bk.is_forbidden(n2, n1)` → `Edge(n1, n2, e1, ARROW)` | $n_2$ cannot cause $n_1$: arrowhead at $n_2$, $n_1$'s mark `e1` kept |
| 125–128 | `bk.is_forbidden(n1, n2)` → `Edge(n1, n2, ARROW, e2)` | the mirror case |

The same function also serves the exogenous rule: `fci_graph` forbids, for example, duration → age at onset, which puts an arrowhead at duration on an age–duration edge (§4.8, example).

**Worked example** ([`ex04_orient_tiers.py`](../guide_examples/s04/ex04_orient_tiers.py)). Four small linear Gaussian systems ($n=5000$), each run with this package's `fci_graph` and with causal-learn left as it is. $L$ is a hidden cause and is never in the table; the suffix gives the tier.

```python
cases["1. a_t <- L -> b_t1 (hidden cause only)"] = np.column_stack([L + e(), L + e()])
cases["2. a_t -> b_t1 (direct effect)"]          = np.column_stack([a, 0.7 * a + e()])
cases["3. B -> A_t -> C_t1 (chain)"]             = np.column_stack([B, A, 0.8 * A + e()])
cases["4. B -> A_t <- L -> C_t1 (hidden cause into a collider)"] = np.column_stack([B, 0.8 * B + L + e(), L + e()])
fci_graph(T, list(range(T.shape[1])), 0.05, -1)                   # tiers 0, 1, (2)
```

```text
1. a_t <- L -> b_t1 (hidden cause only)
   pdx fci_graph       : a_t o-> b_t1
   causal-learn as is  : a_t --> b_t1
2. a_t -> b_t1 (direct effect)
   pdx fci_graph       : a_t o-> b_t1
   causal-learn as is  : a_t --> b_t1
3. B -> A_t -> C_t1 (chain)
   pdx fci_graph       : B o-> A_t; A_t --> C_t1
   causal-learn as is  : B --> A_t; A_t --> C_t1
4. B -> A_t <- L -> C_t1 (hidden cause into a collider)
   pdx fci_graph       : B o-> A_t; A_t --> C_t1
   causal-learn as is  : B --> A_t; A_t --> C_t1
case 4, causal-learn's is_arrow_point_allowed replaced by a mark-only check: B o-> A_t; A_t <-> C_t1
```

Reading the output:
- **Cases 1 and 2.** Two variables cannot tell a hidden cause from a direct effect. The package correctly reports `o->` for both; causal-learn as is claims `-->` for both, including the purely confounded pair.
- **Case 3.** Here the tail at $A_t$ is earned. FCI's rule R1 sees $B{*\!\!\to}A_t\ \text{o}\!\!\to C_{t+1}$ with $B$ and $C_{t+1}$ non-adjacent, and $A_t$ in their separating set, so $A_t$ must be a non-collider: $A_t\to C_{t+1}$.
- **Case 4 shows what the replacement does not fix.** The true PAG is $B\ \text{o}\!\!\to A_t\leftrightarrow C_{t+1}$: $A_t$ is a collider ($B\perp\!\!\!\perp C_{t+1}$ marginally). The package reports $A_t\to C_{t+1}$, a causal claim the data contradict. The cause is a second causal-learn function, `is_arrow_point_allowed` (`FCI.py:294–305`). It refuses an arrowhead at $y$ on $x\ {*\!\!-\!\!*}\ y$ when the knowledge forbids $x\to y$ and $x$ already carries an arrowhead. After `_orient_tiers`, every year-$t+1$ node carries one, so collider orientation can never put an arrowhead at the year-$t$ end. The collider is skipped, and R1 then turns the remaining circle into a tail. With an arrowhead check that looks only at the mark already present, the correct $A_t\leftrightarrow C_{t+1}$ comes back (last line). This matters for `lagged_directed`, the separate graphs' directed statistic and the reported marks; it does not change any adjacency, so the support is unaffected (Notes 4.N, item 1).

**Design choices and pitfalls.**
- `is_required` knowledge is ignored. Nothing in this package adds required edges, but if someone did, the replacement would silently drop them (causal-learn's original handles them).
- If both directions of an edge were forbidden, only the first branch would apply. causal-learn's adjacency search removes such edges before this point, and `fci_graph` never forbids both directions.
- `test_hidden_confounding_keeps_a_circle_at_the_earlier_node` checks case 1 only. Two variables cannot form a collider, so the test cannot see case 4.

### 4.8 `fci_graph` — FCI with tiers, the exogenous rule and the replacement in place

[`step1.py:131`](../../src/pdx/step1.py#L131)

**What it does.** Runs causal-learn's FCI on a table with:
1. one of the two tests above (or any causal-learn test on complete rows);
2. tiered background knowledge;
3. the rule that the `exogenous` columns (age at onset, sex) can be caused by nothing outside `exogenous ∪ exempt` (themselves and era);
4. `_orient_tiers` in place of `fci_orient_bk` for the duration of the call.

It returns causal-learn's endpoint matrix as integers.

**Inputs → outputs.** `table` ($n_{rows}\times m$, NaN allowed for the two package tests), `tiers` (one integer per column), `alpha`, `depth`, `test`, `clusters` (participant per row), `exogenous`, `exempt`, `max_path_length` → $\mathcal G$, an $m\times m$ integer matrix with $\mathcal G[a,b]$ the mark at $a$ on the edge $a$–$b$ (−1 tail, 1 arrowhead, 2 circle, 0 no edge).

**The mathematics: how FCI uses the tests and the knowledge** (causal-learn 0.1.4.8, `causallearn/search/ConstraintBased/FCI.py:1061`).

FCI (Spirtes, Glymour & Scheines 2000; Zhang 2008) assumes the causal Markov condition and faithfulness for the measured variables together with any hidden ones, acyclicity and correct test decisions. It does **not** assume that every common cause is measured. Its phases:

1. **Skeleton by the fast adjacency search** (`causallearn/utils/FAS.py:17`). Start from the complete graph. For depth $\ell=0,1,2,\dots$ up to `depth`, for each adjacent pair $(a,b)$ and each $\mathbf S\subseteq\operatorname{adj}(a)\setminus\{b\}$ with $\lvert\mathbf S\rvert=\ell$, test $a\perp\!\!\!\perp b\mid\mathbf S$. If $p>\alpha$, the edge is marked for removal and $\mathbf S$ joins its separating set. The search is the *stable* variant: removals take effect after all pairs of that depth are tested, so the result does not depend on column order. Two details of this implementation: it keeps testing all sets of the depth after a removal, and the separating set it stores is the **union** of every set that gave $p>\alpha$ at that depth, from both endpoints. Background knowledge enters only to remove an edge whose two directions are both forbidden, which never happens here (tiers and the exogenous rule forbid at most one direction). **So the data decide every adjacency.**
2. **Rule R0** (`FCI.py:308`): every mark becomes a circle. Then the knowledge step runs, which is `_orient_tiers` here: an arrowhead at the later node of every cross-tier edge, and, on an edge between an exogenous column and a column that may not cause it, an arrowhead at the latter. Then unshielded colliders: for $a\ {*\!\!-\!\!*}\ c\ {*\!\!-\!\!*}\ b$ with $a,b$ non-adjacent and $c\notin\operatorname{Sepset}(a,b)$, orient $a{*\!\!\to}c{\leftarrow\!\!*}b$, but only if `is_arrow_point_allowed` permits both arrowheads (§4.7, case 4).
3. **Possible-D-SEP** (`FCI.py:1000`). With hidden variables, a separating set may need nodes that are not adjacent to either endpoint. For each remaining edge, every subset of size at least 2 of Possible-D-SEP$(a,b)$ is tested, except subsets lying entirely within $\operatorname{adj}(a)$ or $\operatorname{adj}(b)$. **`depth` does not cap these sets**, and the knowledge is not consulted. Edges found separable are removed.
4. **R0 again**: all marks back to circles, knowledge, colliders.
5. **Orientation rules until nothing changes**: R1 (away from colliders), R2 (no directed cycles), R3 (double triangle); R4 (discriminating paths, which runs new tests) only in a pass where R1–R3 changed something; then R5–R7 (selection-bias rules) and R8–R10 (circles to tails when an ancestral relation is implied). R1, R2, R3 and R4's collider branch all go through `is_arrow_point_allowed`.

*The tier rule in a two-slice table.* For a panel with covariates and two domains, `step1_table` gives

$$
\underbrace{\text{era}}_{\tau=0}\ \big|\ \underbrace{\text{age},\ \text{sex},\ \text{duration}}_{\tau=1}\ \big|\ \underbrace{Y_{0,t},\ Y_{1,t}}_{\tau=2}\ \big|\ \underbrace{Y_{0,t+1},\ Y_{1,t+1}}_{\tau=3}.
$$

causal-learn's `is_forbidden(a, b)` is true when $\tau(a)>\tau(b)$. So every edge between year $t$ and year $t+1$, or between a baseline variable and a year, gets an arrowhead at its later end in R0. Within a tier the data decide (protocol §6.2), apart from the exogenous rule: `add_forbidden_by_node(nodes[j], nodes[e])` forbids $j\to e$ for each exogenous $e$ and each column $j$ outside `exogenous ∪ exempt`. Beyond the tiers this adds only within-baseline constraints: duration cannot cause age at onset or sex.

Two modelling assumptions sit in the pooling itself: the same transition at every year (rows from years 1→2 and 5→6 are pooled), and conditionally independent rows (rows of one participant are not, which is what the clustered test of §4.6 addresses).

**From the method to the code, line by line.**

| line | code | meaning |
|---|---|---|
| 140–142 | `_register_tests()`; `name = _TESTS.get(test, test)`; `kw = {"clusters": clusters}` | the package tests receive the participant index |
| 143–144 | other tests: `table = table[~isnan(table).any(1)]` | complete rows for causal-learn's own tests |
| 145–146 | `nodes = [GraphNode(f"X{i + 1}") ...]` | causal-learn names its nodes `X1, X2, …` too, and `GraphNode` compares by name, so this knowledge matches FCI's own node objects |
| 147–149 | `bk.add_node_to_tier(nodes[i], int(t))` | tiers |
| 150–154 | `free = exogenous ∪ exempt`; forbid $j\to e$ | the exogenous rule |
| 155–162 | `FCI.fci_orient_bk = _orient_tiers` … `finally: restore` | `rule0` looks the function up in FCI's module namespace at call time, so the swap takes effect inside the call and is undone even after an error |
| 158 | `redirect_stdout(io.StringIO())` | causal-learn prints orientation messages even with `verbose=False` |
| 159–160 | `FCI.fci(table, name, alpha, depth=..., max_path_length=..., background_knowledge=bk, ...)` | the run |
| 163 | `np.asarray(G.graph, int)` | the endpoint matrix |

**Worked example** ([`ex05_fci_graph.py`](../guide_examples/s04/ex05_fci_graph.py)). A hand-made two-slice table ($n=3000$) with a known structure: age at onset → duration → $y_{0,t}$; age → $y_{1,t}$; persistence of both domains; $y_{0,t}\to y_{1,t+1}$. Era and sex are pure noise.

```python
duration = 0.5 * age + e(); y0_t = 0.5 * duration + e(); y1_t = 0.4 * age + e()
y0_t1 = 0.6 * y0_t + e();   y1_t1 = 0.6 * y1_t + 0.3 * y0_t + e()
tiers = [0, 1, 1, 1, 2, 2, 3, 3]               # era | other baseline | year t | year t+1
G = fci_graph(T, tiers, 0.05, 3, "fisherz", exogenous=[1, 2], exempt=[0])
```

```text
endpoint matrix G[i, j] = mark at i (-1 tail, 1 arrowhead, 2 circle):
[[ 0  0  0  0  0  0  0  0]
 [ 0  0  0  2  0  2  0  0]
 [ 0  0  0  0  0  0  2  0]
 [ 0  1  0  0 -1  0  0  0]
 [ 0  0  0  1  0  0 -1 -1]
 [ 0  1  0  0  0  0  0 -1]
 [ 0  0  1  0  1  0  0  0]
 [ 0  0  0  0  1  1  0  0]]
edges, with the exogenous rule:
   age      o-> duration
   age      o-> y1_t
   sex      o-> y0_t1
   duration --> y0_t
   y0_t     --> y0_t1
   y0_t     --> y1_t1
   y1_t     --> y1_t1
lagged adjacency [to, from]:
 [[1 0]
 [1 1]]
edges, without the exogenous rule:
   age      o-o duration
   age      o-> y1_t
   sex      o-> y0_t1
   duration o-> y0_t
   y0_t     --> y0_t1
   y0_t     --> y1_t1
   y1_t     --> y1_t1
```

Reading the output:
- **The endpoint matrix.** Row "duration" (index 3) has −1 in column $y_{0,t}$ (4): a tail at duration. Row $y_{0,t}$ has 1 in column 3: an arrowhead at $y_{0,t}$. Together, duration → $y_{0,t}$.
- **The exogenous rule at work.** With it, the age–duration edge gets an arrowhead at duration. R1 then sees age ${*\!\!\to}$ duration o-> $y_{0,t}$ with age and $y_{0,t}$ non-adjacent (duration separates them), and orients duration → $y_{0,t}$. Without the rule, age o-o duration stays unoriented and duration o-> $y_{0,t}$ keeps its circle.
- **The lagged links** are all correct: both persistences and $y_{0,t}\to y_{1,t+1}$, with tails earned through R1.
- **`sex o-> y0_t1` is a false adjacency.** Sex is pure noise, but its test against $y_{0,t+1}$ rejected at the 5% level for every set tried. With 28 pairs, such an error at $\alpha=0.05$ is not unusual. It is a baseline-to-year edge, so it adds no drift entry.

**Design choices and pitfalls.**
- **`depth` caps only the first phase.** Possible-D-SEP and R4 test larger sets. On a synthetic six-domain table (1,638 rows, 16 columns), FCI with `depth=1` still requested tests with up to 7 conditioning variables. The comment "separating-set cap" in `Step1Config` should say "adjacency-search cap", and the cost per fit depends on more than `depth` (Notes 4.N).
- `max_path_length` limits only R4's discriminating paths; Possible-D-SEP always uses unbounded paths in this version.
- **R4 sometimes does not run.** causal-learn runs it only in a pass where R1–R3 changed a mark (its "first time" exception needs required knowledge, which is never given here). This is a library behaviour, not a package bug, but it means some marks that a complete FCI would orient stay circles.
- **The swap is process-wide.** While `fci_graph` runs, every FCI call in the same process uses `_orient_tiers`. That is safe with the package's process-based parallelism, but would not be under threads.
- The swap fixes the tail of `fci_orient_bk` but not `is_arrow_point_allowed`, so $\leftrightarrow$ can never appear between tiers, and a hidden-confounding collider can come out as $\to$ (§4.7, case 4; Notes 4.N, item 1).

### 4.9 `step1_table` — the two-slice table with the tiers of §6.2

[`step1.py:166`](../../src/pdx/step1.py#L166) (it calls [`data.two_slice_table`](../../src/pdx/data.py#L97), described in §1)

**What it does.** Builds the table FCI runs on and its tiers:
1. `two_slice_table` makes one row per pair of consecutive grid years of one participant: $(z_i,\ [\text{interval}],\ Y_{ik},\ [\text{dose}_{ik},\ \text{dose}_{i,k+1}],\ Y_{i,k+1})$. "Consecutive" means a grid-year difference of exactly 1 when the panel has grid years (a step across a missed year is dropped), and an interval of at most `max_dt` years otherwise. Rows with missing scores are kept.
2. The tiers from `two_slice_table` (baseline 0, year $t$ 1, dose at $t+1$ 2, the rest of year $t+1$ last) are shifted up by one, and era is put back alone in tier 0.
3. With `add_interval`, the interval is partialled out of every other column and dropped.
4. The exogenous columns are those of `exogenous_cols` that exist in $z$.

**Inputs → outputs.** `panel`, `cfg` → `X` (rows × columns, NaN kept), `tiers` (integer per column), `who` (participant index per row), `cols` (column names `b0…`, `y<j>_t`, `dose_t`, `dose_t1`, `y<j>_t1`), `exo` (the exogenous column indices).

**The mathematics.**

*The rows.* For participant $i$ with visits at grid years $\mathrm{yr}_{i1}<\mathrm{yr}_{i2}<\dots$,

$$
\mathcal R=\{(i,k):\ \mathrm{yr}_{i,k+1}-\mathrm{yr}_{ik}=1\},\qquad \text{row}_{(i,k)}=\big(z_i,\ Y_{ik},\ Y_{i,k+1}\big)\in\mathbb R^{p+2d}.
$$

Every row of participant $i$ repeats the same $z_i$; the rows of one participant are the clusters of §4.6.

*Tiers.* With covariates, the final tiers are era 0, the other covariates 1, year $t$ 2, dose at $t+1$ 3 (if included), the rest of year $t+1$ last. Dose at $t$ sits in the year-$t$ tier.

*The interval rule (§6.3).* If fewer than 90% of intervals lie within 12 ± 3 months, the protocol conditions every test on the interval $\Delta_r$. The code partials it out of every column $v$ once, over the rows where $v$ is observed:

$$
v_r\ \leftarrow\ v_r-\hat c_{1,v}\,\Delta_r,\qquad (\hat c_{0,v},\hat c_{1,v})=\arg\min\sum_{r:\,v_r\ \text{obs}}\big(v_r-c_0-c_1\Delta_r\big)^2 .
$$

(The code subtracts the fitted line and adds back $\hat c_{0,v}$, which is the same.) Why this is "conditioning every test on the interval": the projection onto $\operatorname{span}\{1,\Delta,\mathbf S\}$ is the projection onto $\operatorname{span}\{1,\Delta\}$ plus the projection onto $\mathbf S$ residualised on $(1,\Delta)$ (Frisch–Waugh–Lovell). So, on complete rows, the residuals of $U$ and $V$ on $[1,\Delta,\mathbf S]$ equal the residuals of the partialled $U$ and $V$ on $[1,\text{partialled }\mathbf S]$, and the test's $\hat\rho$ is exactly the partial correlation given $\mathbf S\cup\{\Delta\}$. Two approximations remain:
- with missing values, each column is partialled on its own observed rows, not on the rows of the test;
- the Fisher-z statistic uses $\sqrt{n-k-3}$ where conditioning on $\Delta$ as well would give $\sqrt{n-k-4}$. At hundreds of rows this is negligible.

**From the formulas to the code.**

| line | code | meaning |
|---|---|---|
| 170–171 | `base = panel.z if panel.p else None`; `two_slice_table(...)` | the rows; baseline only if the panel has covariates |
| 172–174 | `tiers = tiers + 1`; `tiers[cfg.era_col] = 0` | era alone in the first tier |
| 175–180 | find, extract and delete the `"interval"` column | |
| 181–186 | per column: `lstsq([1, iv], X[ok, k])`; `X[ok, k] - C[ok] @ coef + coef[0]` | the partialling above, on the observed rows (at least 3) |
| 187 | `exo = [c for c in exogenous_cols if base is not None and c < panel.p]` | the exogenous columns; baseline columns come first, so a $z$ index is also a table index |

**Worked example** ([`ex06_step1_table.py`](../guide_examples/s04/ex06_step1_table.py)). Three participants, two domains, four covariates. Participant 1 misses grid year 3 and has a missing score at year 2.

```python
y = [np.array([[0.1, 0.2], [0.3, 0.5], [0.6, 0.9]]),             # participant 0: grid years 1, 2, 3
     np.array([[1.0, 1.1], [1.2, np.nan], [1.9, 2.0]]),           # participant 1: years 1, 2, 4
     np.array([[-0.5, -0.4], [-0.2, -0.3]])]                      # participant 2: years 1, 2
year = [np.array([1, 2, 3]), np.array([1, 2, 4]), np.array([1, 2])]
panel = Panel(y, t, group=[0, 1, 0], z=z, dose=dose, year=year)
X, tiers, who, cols, exo = step1_table(panel, Step1Config())
```

```text
columns: ['b0', 'b1', 'b2', 'b3', 'y0_t', 'y1_t', 'y0_t1', 'y1_t1']
tiers:   [0, 1, 1, 1, 2, 2, 3, 3]   exogenous columns: [1, 2]
row participant: [0, 0, 1, 2]
[[-1.   0.2  0.5 -0.1  0.1  0.2  0.3  0.5]
 [-1.   0.2  0.5 -0.1  0.3  0.5  0.6  0.9]
 [ 0.  -0.3 -0.5  0.4  1.   1.1  1.2  nan]
 [ 1.   0.1  0.  -0.3 -0.5 -0.4 -0.2 -0.3]]

with include_dose: columns ['b0', 'b1', 'b2', 'b3', 'y0_t', 'y1_t', 'dose_t', 'dose_t1', 'y0_t1', 'y1_t1']
                   tiers   [0, 1, 1, 1, 2, 2, 2, 3, 4, 4]

with add_interval: columns ['b0', 'b1', 'b2', 'b3', 'y0_t', 'y1_t', 'y0_t1', 'y1_t1'] (the interval is partialled out, then dropped)
  y0_t: before [ 0.1  0.3  1.  -0.5], after [3.25  2.877 3.577 3.223]; slope on the interval -2.864; after = before - slope x interval: True
  y0_t1: before [ 0.3  0.6  1.2 -0.2], after [3.35  3.095 3.695 3.405]; slope on the interval -2.773; after = before - slope x interval: True
```

Reading the output:
- Four rows: two from participant 0, one from participant 1 (year 2 → 4 is dropped because year 3 is missing), one from participant 2. The row with the missing score is kept.
- The covariates repeat within participant (rows 0 and 1).
- With dose, `dose_t` joins the year-$t$ tier (2) and `dose_t1` gets tier 3, before the rest of year $t+1$ (4).
- With the interval, each column loses its linear trend in the interval (four rows give large slopes here; this is only a demonstration of the arithmetic).

**Design choices and pitfalls.**
- The baseline covariates are partialled on the interval as well. That is what "conditioned on in every test" requires, because tests between a covariate and a score also condition on it.
- The decision whether the interval rule applies is not made here: `run_hp_real.py` takes it as a flag (`--interval-in-step1`), set from WP0's report.
- `max_dt` is ignored whenever the panel has grid years; PPMI panels and the family-A simulator both have them.
- Pooling transitions from all years assumes one transition structure for every year of follow-up.

### 4.10 `_lagged_index` and `lagged_adjacency` — from the graph to the support's [to, from] matrix

[`step1.py:191`](../../src/pdx/step1.py#L191), [`step1.py:195`](../../src/pdx/step1.py#L195)

**What it does.** `_lagged_index` finds the column positions of `y<j>_t` and `y<j>_t1` for each domain. `lagged_adjacency` reads, for every pair of domains, whether domain $k$ in year $t$ and domain $j$ in year $t+1$ are adjacent, with any marks.

**Inputs → outputs.** $\mathcal G$, `cols`, $d$ → a boolean $d\times d$ matrix $L$ in the **[to, from]** convention.

**The mathematics.**

$$
L_{jk}=\mathbf 1\big\{\mathcal G[\,y_{k,t},\ y_{j,t+1}\,]\neq0\big\},\qquad j=\text{to},\ k=\text{from}.
$$

The row index is the target, as in the drift: $A_{jk}$ is the effect of domain $k$ on the rate of change of domain $j$. So $L_{jk}=1$ proposes the drift entry $A_{jk}$. The diagonal $L_{jj}$ is the adjacency between a domain and itself a year later (persistence).

Any marks count: $\text{o}\!\!\to$, $\to$ and (if it could occur) $\leftrightarrow$ all give $L_{jk}=1$. This follows protocol §6.3: entries whose adjacency may reflect hidden confounding stay in the support, so that the null model is not misspecified. Same-year adjacencies and baseline–year adjacencies are not read at all.

**From the formula to the code.** `graph[it[k], it1[j]] != 0` is $\mathcal G[y_{k,t}, y_{j,t+1}]\neq0$. The comprehension's outer loop runs over $j$ (rows) and its inner loop over $k$ (columns). Reading the mark at $y_{k,t}$ suffices, because causal-learn stores a non-zero code at both ends of every edge.

**Worked example** ([`ex07_lagged.py`](../guide_examples/s04/ex07_lagged.py)). A hand-made endpoint matrix for `cols = ["b0", "y0_t", "y1_t", "y0_t1", "y1_t1"]`:

```python
edge("y0_t", "y0_t1", 2, 1)                     # y0_t  o-> y0_t1  (persistence of domain 0)
edge("y0_t", "y1_t1", -1, 1)                    # y0_t  --> y1_t1  (domain 0 drives domain 1, oriented)
edge("y1_t", "y1_t1", 2, 1)                     # y1_t  o-> y1_t1
edge("y0_t1", "y1_t1", 2, 2)                    # same-year o-o: adds no drift entry
edge("b0", "y1_t", 2, 1)                        # baseline o-> year t: adds no drift entry
print(lagged_adjacency(G, cols, 2).astype(int)); print(lagged_directed(G, cols, 2).astype(int))
```

```text
lagged_adjacency [to, from]:
 [[1 0]
 [1 1]]
lagged_directed  [to, from]:
 [[0 0]
 [1 0]]
```

Row 1, column 0 is "domain 0 → domain 1": the entry $A_{10}$. The same-year and baseline edges leave no trace.

**Design choices and pitfalls.**
- The yearly map is not the drift. A two-slice adjacency $y_{k,t}$ – $y_{j,t+1}$ also arises from paths that run *within* the year through other domains ($k\to m\to j$ in continuous time makes the yearly transition entry $[e^{A}]_{jk}$ non-zero even if $A_{jk}=0$). The support can therefore contain entries that are truly zero in the drift; the engine estimates them, which costs power but not validity.
- Lagged adjacencies are read from year-$t$ to year-$t+1$ columns only, never in the other direction, so no time-reversed entry can enter.

### 4.11 `lagged_directed` — the lagged edges oriented from year t to year t+1

[`step1.py:201`](../../src/pdx/step1.py#L201)

**What it does.** Like `lagged_adjacency`, but only for edges with a **tail** at the year-$t$ node and an arrowhead at the year-$t+1$ node.

**Inputs → outputs.** $\mathcal G$, `cols`, $d$ → boolean $d\times d$, [to, from].

**The mathematics.**

$$
\vec L_{jk}=\mathbf 1\big\{\mathcal G[y_{k,t},y_{j,t+1}]=-1\ \wedge\ \mathcal G[y_{j,t+1},y_{k,t}]=1\big\}.
$$

In a PAG this is the claim "$y_{k,t}$ is an ancestor of $y_{j,t+1}$", that is, a causal (possibly indirect) effect. Because of the tiers, the arrowhead at the year-$t+1$ end is always present; the information is in the tail.

**Worked example.** In §4.10's example only $y_{0,t}\to y_{1,t+1}$ has a tail, so only $\vec L_{10}=1$.

**Design choices and pitfalls.**
- It is used only by `separate.py` (the "directed" statistic). The support never uses it.
- Tails at the year-$t$ end can be wrong when a hidden cause meets a collider at the year-$t$ node (§4.7, case 4), so $\vec L$ can overstate causal orientation (Notes 4.N, item 1).

### 4.12 `run_fci` — one Step 1 fit on a panel

[`step1.py:208`](../../src/pdx/step1.py#L208)

**What it does.** Builds the table (`step1_table`), sets `exempt = [era_col]` when the panel has covariates, runs `fci_graph`, and returns the graph with counts.

**Inputs → outputs.** `panel`, `cfg` → (`G`, `cols`, number of rows, number of distinct participants, number of complete rows).

**The mathematics.** None beyond §4.8–§4.9. The three counts are aggregates, safe to report.

**Worked example** ([`ex08_select_support.py`](../guide_examples/s04/ex08_select_support.py), first part). Family-A scenario S1 (one mechanism; PIGD twice as fast), four domains:

```python
panel, info = scenarios.generate("S1", 150, np.random.default_rng(1), d=4)
G, cols, n_rows, n_part, n_complete = run_fci(panel, Step1Config())
```

```text
run_fci: 587 rows from 139 participants, 428 complete rows, 0.70 s
```

(150 simulated participants, 139 left after the phenotype rule removed the indeterminate ones.) Test-wise deletion uses the 587 rows; complete-case analysis would have only 428.

**Design choices and pitfalls.** `exempt` is set only when the panel has covariates, which is when era exists. With `era_col=None` era would sit in tier 1 like the other covariates and would also be barred from causing age at onset and sex.

### 4.13 `select_support` — the support of Steps 2 to 5

[`step1.py:216`](../../src/pdx/step1.py#L216)

**What it does.** Runs Step 1 and returns the support: the lagged adjacencies found, together with the expert map's links and every own rate.

**Inputs → outputs.** `panel`, `expert` (boolean $d\times d$ [to, from], or None), `cfg` (default: the fast configuration) → boolean $d\times d$ [to, from], ready for `ModelSpec.with_support`.

**The mathematics.**

$$
\mathcal S\;=\;\underbrace{\{(j,k):L_{jk}=1\}}_{\text{Step 1, fast configuration}}\ \cup\ \underbrace{\{(j,k):\mathcal E_{jk}=1\}}_{\text{expert map (WP1)}}\ \cup\ \underbrace{\{(j,j)\}}_{\text{own rates}} .
$$

Its size $n_{sup}$ sets H-P's degrees of freedom, $(G-1)(n_{sup}-1)$ (§0.1). Taking the union means a link enters if *either* source proposes it ("every link any group might use", §6.4). Missing a true link would misspecify the null model; including a zero link only costs power.

**From the formula to the code.** Line 220 runs FCI, line 221 reads $L$, line 223 adds the expert map with `|`, line 224 adds the identity.

**Worked example** ([`ex08_select_support.py`](../guide_examples/s04/ex08_select_support.py)). The true drift support of the four-domain simulator is tremor → function, axial → function, function → independence, plus the own rates. The expert map given to every scenario adds the candidate independence ← axial, which is zero in S1.

```python
truth, expert = family_a.true_support(info["truth"]), info["support"]
select_support(panel)                 # Step 1 alone (plus own rates)
select_support(panel, expert)         # the support of Steps 2 to 5
```

```text
true drift support [to, from]:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 0 1 1]]
expert map handed to the analysis (truth + candidate independence <- axial):
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
lagged adjacencies found [to, from]:
 [[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 1 1]]
select_support without the map:
 [[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 1 1]]
select_support with the map (the support of Steps 2-5):
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
make_support_fn gives the same support: True

544 participants (0.09 s): lagged adjacencies found [to, from]:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 0 1 1]]
```

Reading the output:
- With 139 participants, Step 1 finds only function → independence among the cross-links; the two weaker links into function (drift 0.10 and 0.20 per year) are missed. The expert map restores them, which is why the union is taken.
- With 544 participants Step 1 alone recovers exactly the true drift support.
- The support with the map equals the map here, because Step 1 found nothing outside it.

**Design choices and pitfalls.**
- The diagonal is always in, whatever FCI finds: every domain has an own rate in the engine.
- `cfg` defaults to the fast configuration. The support of the real analysis is therefore selected with the uncorrected test, which keeps too many adjacencies when rows are clustered (§4.6). That errs on the side of a larger support, the safe side for the null model.

### 4.14 `make_support_fn` — the same selection rule in every bootstrap replicate

[`step1.py:227`](../../src/pdx/step1.py#L227)

**What it does.** Returns a function of one argument, `panel → select_support(panel, expert, cfg)`, with the expert map and the configuration fixed.

**Inputs → outputs.** `expert`, `cfg` → a callable.

**The mathematics.** H-P's statistic is computed after a data-dependent choice of support. Its bootstrap reference is valid only if every replicate repeats that choice on its own simulated data (§7.3.1, step 3):

$$
\mathrm{LR}^*_b=\mathrm{LR}\big(\text{panel}^*_b;\ \hat{\mathcal S}(\text{panel}^*_b)\big),\qquad \hat{\mathcal S}=\texttt{make\_support\_fn(expert, cfg)}.
$$

**Worked example.** The line `make_support_fn gives the same support: True` in §4.13.

**Design choices and pitfalls.**
- The callable is built where it is used: in each worker's setup (`hp.boot_setup`, `speed.sample_setup`) and inside the job functions of `qualification.py` and `gate.py`. So the lambda never has to be pickled for the `spawn` workers.
- A simulated replicate keeps the observed panel's grid years and missingness (`engine.simulate.simulate_from_fit` returns `panel.with_y(...)`), so the replicate's table has the same rows as the observed one.

### 4.15 `reported_map` and `_rename` — the reported map with its marks

[`step1.py:233`](../../src/pdx/step1.py#L233), [`step1.py:252`](../../src/pdx/step1.py#L252)

**What it does.** Runs one Step 1 fit (default: the clustered test, cap 4) and returns aggregate output: the column names (with domain names if given), every edge with its marks, the counts, the lagged adjacency and the configuration.

**Inputs → outputs.** `panel`, `cfg`, `names` (domain names) → a JSON-ready dictionary.

**The mathematics.** The edge string for columns $a<b$ is (mark at $a$, with `>` turned into `<`) + `-` + (mark at $b$). So $\mathcal G[a,b]=2,\ \mathcal G[b,a]=1$ prints `o->`; $(-1,1)$ prints `-->`; $(1,1)$ prints `<->`; $(2,2)$ prints `o-o`. `_rename` turns `y3_t1` into `<names[3]>_t1` and leaves the covariates as `b0`…`b3`.

**Worked example** ([`ex09_reported_map.py`](../guide_examples/s04/ex09_reported_map.py)). Scenario S1, 544 participants, the default configuration:

```python
out = reported_map(panel, names=panel.domains)
```

```text
0.74 s; keys: ['columns', 'edges', 'rows', 'complete_rows', 'participants', 'lagged_adjacency', 'config']
rows 2349 | complete rows 1666 | participants 544
config: {"alpha": 0.05, "depth": 4, "test": "cluster_fisherz", "era_col": 0, "exogenous_cols": [1, 2], "add_interval": false, "include_dose": false, "max_dt": 1.5, "max_path_length": -1}
edges between year t and year t+1, and within year t+1:
   tremor_t         o-> tremor_t1
   tremor_t         o-> function_t1
   axial_t          o-> axial_t1
   axial_t          o-> function_t1
   function_t       o-> function_t1
   function_t       --> independence_t1
   independence_t   o-> independence_t1
   axial_t1         o-> function_t1
   function_t1      --> independence_t1
lagged_adjacency [to, from]: [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]
```

The lagged adjacencies match the true drift support. Most lagged edges keep a circle at year $t$: in this simulator every participant has their own speed and load, hidden causes of all their scores, so the data cannot exclude confounding. One lagged edge (function → independence) and one same-year edge got tails from the orientation rules (R1–R10).

**Design choices and pitfalls.**
- `rows`, `complete_rows` and `participants` are counts; the edges are a graph. Nothing participant-level leaves the function.
- `reported_map` and the support use different configurations (clustered test, cap 4, against the fast test, cap 3), so the reported map's lagged adjacencies need not equal the support's Step 1 part.
- The marks inherit the limitation of §4.7: no `<->` can appear between tiers, and a `-->` may stand for a hidden-confounding collider.

### 4.16 `sortability` — the variance and R² sortability control

[`step1.py:259`](../../src/pdx/step1.py#L259)

**What it does.** For a table and a set of directed edges between its columns, it returns the share of ancestor–descendant pairs whose score (variance, or $R^2$) increases along the causal order. A value near 1 means a method that simply sorts variables by that score would recover the order (Reisach 2021, 2023), so accuracy on such data says little about a causal-discovery method.

**Inputs → outputs.** `table` (complete rows are used), `directed` (boolean, **[from, to]**: `directed[a, b]` = $a\to b$), `kind` (`"var"` or anything else for $R^2$) → a number in $[0,1]$, or NaN if there are no pairs.

**The mathematics.** Let $\operatorname{De}(a)$ be the descendants of $a$ under `directed`. The transitive closure is computed by Warshall's algorithm, $R\leftarrow R\vee(R_{\cdot k}\wedge R_{k\cdot})$ for $k=1,\dots,m$. Then, with the score $s_a=\operatorname{Var}(v_a)$ or $s_a=R^2_a$, the $R^2$ of $v_a$ regressed on all other columns with an intercept,

$$
\text{sortability}=\frac{1}{\lvert\mathcal P\rvert}\sum_{(a,b)\in\mathcal P}\Big(\mathbf 1\{s_a<s_b\}+\tfrac12\mathbf 1\{s_a=s_b\}\Big),\qquad \mathcal P=\{(a,b):b\in\operatorname{De}(a)\}.
$$

**Worked example** ([`ex10_sortability.py`](../guide_examples/s04/ex10_sortability.py)). A chain $x_0\to x_1\to x_2$ with unit coefficients, so the raw variances are 1, 2, 3. The columns are shuffled to show that order does not matter.

```python
raw = np.column_stack([x2, x0, x1])                # columns shuffled on purpose: 0 = x2, 1 = x0, 2 = x1
directed[1, 2] = directed[2, 0] = True             # x0 -> x1 -> x2 in the usual [from, to] convention
std = (raw - raw.mean(0)) / raw.std(0)
```

```text
variances, raw: [3.01 1.   1.99]
variances, standardised (all 1 up to rounding): ['1.00000000000000178', '1.00000000000000666', '1.00000000000000377']
raw           var-sortability = 1.000, R2-sortability = 0.667
standardised  var-sortability = 0.000, R2-sortability = 0.667
```

Reading the output:
- Raw, variance rises along the chain: var-sortability 1.
- $R^2$ is scale-free, so it is the same raw and standardised. In theory $R^2_{x_0}=0.5$, $R^2_{x_1}=0.75$, $R^2_{x_2}=2/3$ (0.49, 0.75 and 0.67 in this sample): two of the three pairs increase, 0.667.
- **Standardised var-sortability should be 0.5 (all ties) but prints 0.** After standardisation the variances differ only by rounding (last digits above), and the code's exact `a == b` test treats rounding noise as order.

**Design choices and pitfalls.**
- Each ancestor pair counts once. Reisach et al. count a pair once *per path length* that connects it (their code multiplies the edge matrix by itself), and treat values within a relative $10^{-9}$ as ties. On graphs with several paths between a pair the two definitions differ, so published values are not directly comparable (Notes 4.N).
- This function computes the sortability scores only. The protocol's "sortability baseline that ignores conditional independence" (§7.2), a learner that orders by variance, is not in the code.
- `directed` uses [from, to], unlike every other matrix in this chapter.

### 4.17 `separate.py`, the module docstring — separate graphs, the transparent reference

[`separate.py:1`](../../src/pdx/separate.py#L1)

**What it does.** States the procedure of protocol §7.3.5 (v0.3's Analysis B):
- each phenotype is subsampled `n_sub` times to the same size, 90% of the smaller group;
- the Step 1 learner, in its fast configuration, runs on every subsample, giving each lagged link's selection frequency in each phenotype;
- the statistic is $D(e)=\hat p_{TD}(e)-\hat p_{PIGD}(e)$, for adjacencies and for directed edges separately;
- the references are permuted labels (by participant) and, optionally, the definition-matched generator's datasets;
- a link is a candidate if its standardised $D(e)$ exceeds a family-wise threshold (the 95th percentile of the maximum over links in each null dataset; Westfall & Young 1993) **and** $\lvert D(e)\rvert\ge0.25$ (provisional, D10).

The last sentence is the important one: **never a mechanism verdict**. A pure speed difference also changes yearly graphs, so separate graphs show clinicians *where* the yearly graphs differ, not *why*.

**Inputs → outputs.** None (a docstring); the functions follow in §4.18–§4.20.

**The mathematics.** §4.18 (frequencies), §4.19 (the difference) and §4.21 (the decision rule).

**Worked example.** §4.20 (a real run) and §4.21 (the decision rule on made-up numbers).

**Design choices and pitfalls.** Equal subsample sizes remove the most obvious artefact of naive separate graphs: a larger group has more power, so it finds more links (C11 in §4.29 shows this artefact).

### 4.18 `lagged_frequencies` — how often each lagged link is selected

[`separate.py:21`](../../src/pdx/separate.py#L21)

**What it does.** Draws `n_sub` subsamples of `size` participants without replacement from the index set `idx`, runs `run_fci` on each, and averages the lagged adjacency and directed matrices.

**Inputs → outputs.** `panel`, `idx` (participants of one group, or of one permuted group), `n_sub`, `size`, `rng`, `cfg` → two $d\times d$ matrices of frequencies in $[0,1]$, [to, from].

**The mathematics.** For subsamples $b=1,\dots,B$ ($B=$ `n_sub`) of group $g$,

$$
\hat p_g(e)=\frac1B\sum_{b=1}^{B}\mathbf 1\{e\in\hat L^{(g,b)}\},\qquad e=(j,k),
$$

and the same with $\vec L$ for directed edges. It estimates the probability that the learner selects link $e$ on a dataset of `size` participants from group $g$.

**Worked example.** Part 2 of [`ex11_separate.py`](../guide_examples/s04/ex11_separate.py) (§4.20) runs it 30 times.

**Design choices and pitfalls.** Each subsample is a `Panel.subset`, so the subsample keeps its participants' grid years and missingness, and FCI clusters by subsample participant.

### 4.19 `_difference` — the observed (or permuted) difference of frequencies

[`separate.py:33`](../../src/pdx/separate.py#L33)

**What it does.** Splits the participants by a label vector (the real phenotypes or a permutation), sets the subsample size, and returns TD's frequencies minus PIGD's.

**Inputs → outputs.** `panel`, `group` (0/1 per participant), `n_sub`, `frac`, `rng`, `cfg` → $(D_{adj}, D_{dir})$, each $d\times d$ [to, from].

**The mathematics.**

$$
m=\max\big(\lfloor \text{frac}\cdot\min(n_0,n_1)\rfloor,\ 2\big),\qquad D(e)=\hat p_0(e)-\hat p_1(e)\in[-1,1].
$$

$D(e)>0$ means the link is selected more often in TD (group 0).

**Worked example.** See §4.20.

**Design choices and pitfalls.** The smaller group's subsamples share 90% of its participants, so they are nearly the same dataset: that group's frequencies tend to sit near 0 or 1. The larger group's subsamples overlap less. The permutation reference keeps the group sizes, so it reproduces this asymmetry.

### 4.20 `separate_graph_reference` — the statistic, two references and the decision

[`separate.py:41`](../../src/pdx/separate.py#L41)

**What it does.**
1. Computes the observed $D_{adj}$ and $D_{dir}$ with the real labels.
2. Repeats `_difference` with `n_perm` permutations of the participants' labels.
3. Applies the decision rule (`decide`, §4.21) to the permutation reference.
4. If `null_panels` is given (datasets from the definition-matched generator, §7.3.6), computes $D$ on each with its own labels and applies the rule to that reference too.

**Inputs → outputs.** `panel`, `n_sub` (25), `frac` (0.9), `n_perm` (50), `rng`, `cfg` (fast), `threshold` (0.25), `alpha_fw` (0.05), `null_panels` → a dictionary with `D_adjacency`, `D_directed`, `n_sub`, `threshold_abs`, `permutation_reference` and, if requested, `definition_matched_reference`. Each reference holds, for `adjacency` and `directed`, the standardised threshold and a 0/1 candidate matrix.

**The mathematics.** The permutation reference: under the null that TD and PIGD participants are exchangeable, every relabelling $\pi$ of participants is equally likely, and $D^{*}_{\pi}(e)$ is a draw of $D(e)$ under that null. Labels are permuted per participant, so the clusters stay intact. The definition-matched reference replaces "no difference at all" with "the artefacts of the phenotype rule plus a fitted speed difference, but one mechanism"; the gap between the two references is what the protocol calls the artefact of defining phenotypes from graph variables.

**From the method to the code.**

| line | code | meaning |
|---|---|---|
| 47 | `_difference(panel, panel.group, ...)` | observed $D$ |
| 62–66 | `_difference(panel, rng.permutation(panel.group), ...)` | $D^*_\pi$, `n_perm` times |
| 67 | `decide(perm_adj, perm_dir, "permutation_reference")` | §4.21 |
| 68–74 | for each generator panel, `_difference(p, p.group, ...)` | the definition-matched reference |

**Worked example** ([`ex11_separate.py`](../guide_examples/s04/ex11_separate.py), part 2). Scenario S1, small settings so that it runs in seconds:

```python
panel, _ = scenarios.generate("S1", 300, np.random.default_rng(4), d=4)
res = separate.separate_graph_reference(panel, n_sub=3, n_perm=4, rng=np.random.default_rng(1),
                                        cfg=Step1Config(depth=2))
```

```text
real run, 264 participants (TD 130, PIGD 134), 30 FCI fits, 1.7 s
D_adjacency [to, from]:
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [1. 0. 0. 0.]]
D_directed  [to, from]:
 [[-0.333  0.     0.     0.   ]
 [ 0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.   ]]
adjacency: threshold 1.50, candidates [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
directed: threshold 1.45, candidates [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

Reading the output:
- The fits: $2\times3$ subsamples for the observed labels plus $2\times3\times4$ for the permutations, 30 in all.
- Independence ← tremor was selected in all three TD subsamples and in none of the PIGD ones, $D=1$, and it passes both parts of the rule. It is not a mechanism difference: both phenotypes share one drift, in which tremor reaches independence within a year through function (§4.10), so the yearly link exists in both, and whether FCI detects it in a given sample is partly chance. With 90% subsamples the three TD fits are nearly the same dataset, so one chance detection repeats three times; four permutations give a very rough threshold. This is a demonstration of the arithmetic at toy settings; it also shows why a candidate is "where yearly graphs differ", never a mechanism verdict.
- The protocol's settings (25 subsamples, 50 permutations) cost $(1+50)\times2\times25=2{,}550$ FCI fits per analysis, plus $2\times25=50$ for each generator dataset.

**Design choices and pitfalls.**
- The observed $D$ and the permutations draw from the same random stream, so the observed statistic itself carries Monte Carlo error from its 25 subsamples.
- The permutation null is "TD and PIGD identical". A pure speed difference violates it, so candidates are expected under speed alone; the definition-matched reference is the one that can separate artefact from difference.
- `D_directed` inherits the false tails of §4.7, case 4.

### 4.21 `decide` — standardisation, the max-T threshold and the candidates

[`separate.py:51`](../../src/pdx/separate.py#L51)

**What it does.** For one reference (a stack of null $D$ matrices) and each kind (adjacency, directed): standardises every link by its null mean and spread, takes the maximum over links in each null dataset, uses the $(1-\alpha_{fw})$ quantile of those maxima as the threshold, and flags the links whose standardised observed $D$ exceeds it and whose raw $\lvert D\rvert$ is at least 0.25.

**Inputs → outputs.** `null_adj`, `null_dir` (arrays $n_{null}\times d\times d$), `label` → writes `{kind: {"fw_threshold_standardised": c, "candidates": ...}}` into `out[label]`.

**The mathematics.** For each link $e$, over the null datasets $\pi=1,\dots,P$:

$$
\mu_e=\frac1P\sum_\pi D^*_\pi(e),\qquad \sigma_e=\max\Big(\operatorname{sd}_\pi D^*_\pi(e),\ \tfrac{1}{n_{sub}}\Big),\qquad T_e=\frac{\lvert D(e)-\mu_e\rvert}{\sigma_e},
$$

$$
c=\text{quantile}_{1-\alpha_{fw}}\Big\{\max_{e}\frac{\lvert D^*_\pi(e)-\mu_e\rvert}{\sigma_e}\Big\}_{\pi=1}^{P},\qquad
\text{candidate}(e)\iff T_e>c\ \wedge\ \lvert D(e)\rvert\ge0.25 .
$$

This is Westfall and Young's single-step max-T. If no link truly differs, $P(\max_e T_e>c)\approx\alpha_{fw}$, so the probability of *any* false candidate is controlled. Links with different null spreads are put on one scale first. The floor $1/n_{sub}$, the smallest non-zero change of a frequency, keeps a link never selected in the null (spread 0) from getting an infinite $T_e$. The raw threshold 0.25 adds a minimum size, so that a tiny but consistent difference is not reported.

**From the formulas to the code.** Line 54 `mu = N.mean(0)`; line 55 the floored `sd` with `ddof=1`; line 56 `tmax`, the maximum of the standardised null over all $d^2$ links; line 57 `np.quantile(tmax, 1 - alpha_fw)` (numpy's default linear interpolation); line 58 the two conditions.

**Worked example** ([`ex11_separate.py`](../guide_examples/s04/ex11_separate.py), part 1). `_difference` is temporarily replaced by a queue of made-up matrices ($d=2$, $n_{sub}=10$, 20 permutations), so the arithmetic can be followed without FCI:

```python
observed = np.array([[0.0, 0.1], [0.5, 0.0]])          # D(e) = p(e, TD) - p(e, PIGD), [to, from]
null = [np.clip(np.round(rng.normal(0, [[0.02, 0.1], [0.15, 0.02]]), 1), -1, 1) for _ in range(20)]
separate._difference = lambda *args: queue.pop(0)        # observed first, then the 20 null matrices
out = separate.separate_graph_reference(dummy, n_sub=10, n_perm=20, rng=rng)
```

```text
null mean:
 [[0.005 0.015]
 [0.035 0.   ]] 
null spread (floored at 1/n_sub = 0.1):
 [[0.1   0.1  ]
 [0.169 0.1  ]]
standardised observed |D - mu| / sd:
 [[0.05  0.85 ]
 [2.744 0.   ]]
max over links in each null dataset: [0.38 0.38 0.38 0.8  0.85 0.85 0.85 0.85 0.97 0.97 0.97 1.15 1.39 1.39
 1.39 1.39 1.56 1.56 1.85 2.15]
95th percentile by hand: 1.865; code: 1.865
candidates (beyond the threshold and |D| >= 0.25): [[0, 0], [1, 0]]
```

Reading the output:
- Three of the four spreads are floored at 0.1; only the link "0 → 1" (row 1, column 0) varies more in the null (0.169).
- That link's observed $D=0.5$ standardises to $(0.5-0.035)/0.169=2.744$, above the threshold 1.865, and $0.5\ge0.25$: a candidate.
- Link "1 → 0" ($D=0.1$) gives 0.85: below the threshold, and below 0.25 anyway.
- The threshold is the interpolated 95th percentile of 20 maxima (between 1.85 and 2.15); with 50 permutations it is the same kind of estimate, from 50 values.

**Design choices and pitfalls.**
- The maximum runs over all $d^2$ links, own lags included. Own lags are selected in nearly every subsample of both groups, so their null spread is floored and they rarely drive the maximum.
- The null mean and spread come from the same null datasets that give the maxima. That is the usual practice; with 50 datasets the threshold has noticeable Monte Carlo error.
- Protocol §7.3.5 says "standardised by its spread"; the code also centres by the null mean. Under permutation the mean is near zero; under the definition-matched reference it is not, and centring then asks the intended question: a difference beyond what the generator produces.

### 4.22 `REGISTRY` — the named comparators and their status

[`comparators/__init__.py:9`](../../src/pdx/comparators/__init__.py#L9)

**What it does.** Records, for each comparator C1–C12 of protocol §8.4, its name, its software and its status in this repository. The docstring states the comparators' role: a descriptive benchmark of how often their output would be read as a mechanism difference, not part of any decision.

| status | comparators |
|---|---|
| implemented here | C3 (`c3_nct`), C5 (`c5_tsldiffpc`; the cross-check against pyCIPHOD needs installation), C11 (`c11_naive`), C12 (`c12_dynotears`) |
| available in the environment, wrapper to be written in WP3 | C7 (CD-NOD with time tiers), C10 (joint causal inference; J-PCMCI+) |
| needs software installed with the investigator's approval (D23) | C1 (multigroup RI-CLPM, lavaan), C2 (panel GVAR and IVPP), C4 (naive multigroup continuous-time model), C6 (DCI), C8 (sparse mechanism shift), C9 (data-shared lasso; multi-VAR) |

**Inputs → outputs.** A constant dictionary.

**The mathematics.** None.

**Worked example.** `tests/test_comparators.py` checks that the keys are exactly C1–C12.

**Design choices and pitfalls.** The speed-matched arm of §8.4 (C5 and C11 against a TD reference simulated at each PIGD patient's speed) is not in the code yet; nor is any harness that runs the comparators on the qualification datasets (only `scripts/smoke_test.py` calls them).

### 4.23 `comparators/run.py`, the docstring and `_pairs` — consecutive complete visit pairs

[`run.py:1`](../../src/pdx/comparators/run.py#L1), [`run.py:19`](../../src/pdx/comparators/run.py#L19)

**What it does.** The docstring: each wrapper returns the links it would report as differing between the phenotypes, in the engine's [to, from] convention, with own lags (persistence) reported separately from cross-links, so the benchmark can score them against the true speed-adjusted classes.

`_pairs` collects, for one phenotype or for all participants, every pair of consecutive visits at most `max_dt` years apart with **no missing domain at either visit**.

**Inputs → outputs.** `panel`, `g` (0, 1 or None for all), `max_dt` (1.5) → `X` ($n_{pairs}\times d$, year $t$), `X1` (year $t+1$), `who` (participant per pair).

**The mathematics.**

$$
\mathcal P_g=\{(i,k):\ g_i=g,\ t_{i,k+1}-t_{ik}\le1.5,\ Y_{ik}\ \text{and}\ Y_{i,k+1}\ \text{complete}\}.
$$

**Worked example** ([`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py), first part; scenario S1, four domains):

```python
panel, _ = scenarios.generate("S1", 600, np.random.default_rng(3), d=4)
X, X1, who = comp._pairs(panel)
both = comp._standardiser(panel)(np.vstack([X, X1]))
```

```text
_pairs: 1702 complete consecutive pairs from 528 participants
pooled standardised x_t and x_t+1: means [-0.  0.  0. -0.] SDs [1. 1. 1. 1.]
   phenotype 0: 856 pairs
   phenotype 1: 846 pairs
```

**Design choices and pitfalls.**
- This is **not** the row set of Step 1: Step 1 uses grid years (a difference of exactly 1) and keeps incomplete rows for test-wise deletion; `_pairs` uses the time rule and complete cases. C11 mixes both (its separate graphs use Step 1, its interaction tests use `_pairs`). For a comparator benchmark this is a design choice, but it should be stated (Notes 4.N).
- Pooling the pairs treats them as independent rows, as C5 and C12 assume.

### 4.24 `_standardiser` — one scale for both phenotypes

[`run.py:33`](../../src/pdx/comparators/run.py#L33)

**What it does.** Computes each domain's mean and SD over all complete pairs of both phenotypes, both slices stacked, and returns a function that standardises any array with those constants.

**Inputs → outputs.** `panel` → a function `A ↦ (A − μ) / σ`.

**The mathematics.** $\mu_j$ and $\sigma_j$ over $\{x_{t,j}\}\cup\{x_{t+1,j}\}$ of all pairs. Both phenotypes and both slices share one scale, so differences in scale between the phenotypes stay in the data (a faster phenotype has larger yearly changes), as they would in a real application.

**Worked example.** §4.23: the pooled standardised values have means 0 and SDs 1.

**Design choices and pitfalls.** Standardising per phenotype would remove part of the speed signal from the comparators' input and change what they are benchmarked on. Standardising at all matters for C12: DYNOTEARS on raw simulated scales can exploit variance differences between variables (Reisach 2021), which standardised scores remove (protocol §7.2, methods table).

### 4.25 `_split` — cross-links and own lags apart

[`run.py:40`](../../src/pdx/comparators/run.py#L40)

**What it does.** Splits a boolean $d\times d$ [to, from] matrix into its off-diagonal part (cross-links, as a 0/1 matrix) and its diagonal (own lags, as a 0/1 list).

**Inputs → outputs.** `M` → `{"cross_links": [[...]], "self_lags": [...]}`.

**The mathematics.** None.

**Worked example** (from [`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py)):

```python
comp._split(np.array([[True, True], [False, True]]))     # [to, from]: link 1 -> 0, and both own lags
```

```text
_split example: {'cross_links': [[0, 1], [0, 0]], 'self_lags': [1, 1]}
```

**Design choices and pitfalls.** Own lags are separated because speed changes persistence the most (§0.5, idea 1). A comparator that flags persistence under speed alone is reading speed as mechanism, and the benchmark scores that separately (protocol §8.5).

### 4.26 `c12_dynotears` — C12: one DYNOTEARS graph per phenotype, compared link by link

[`run.py:47`](../../src/pdx/comparators/run.py#L47)

**What it does.** For each phenotype, standardises its pairs, fits DYNOTEARS (§4.30–§4.35) with the current slice $x_{t+1}$ and the lagged slice $x_t$, and keeps the non-zero pattern of the lagged and same-time weights. Links present in one phenotype's graph and absent from the other's are reported as differing.

**Inputs → outputs.** `panel`, `lam` (0.05), `tau_w` (0.3), `tau_a` (0.1) → `cross_links`, `self_lags` (discordant lagged links) and `same_time_discordant` (the number of discordant same-time edges).

**The mathematics.** With $\hat A_g$ and $\hat W_g$ from DYNOTEARS ([from, to]),

$$
\text{disc}_{jk}=\mathbf 1\{[\hat A_0]_{kj}\neq0\}\ \oplus\ \mathbf 1\{[\hat A_1]_{kj}\neq0\},
$$

where $\oplus$ is "exactly one of the two". Only the supports are compared, not the sizes. The settings are the paper's large-sample ones (protocol §8.4).

**From the formula to the code.** Line 54 `DY.dynotears(std(X1), std(X), lam, lam, tau_w, tau_a)`: DYNOTEARS' "X" is the current slice ($x_{t+1}$) and its "Y" the lagged one ($x_t$). Line 55 `A.T != 0` turns [from, to] into [to, from]. Line 57 `lagged[0] != lagged[1]` is the exclusive or.

**Worked example** ([`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py); six domains, 800 TD and 250 PIGD, no mechanism difference). The lines for C12:

```text
=== S0: 1050 participants; domains ['tremor', 'axial', 'brady', 'function', 'independence', 'cognition']
C12 DYNOTEARS  (0.0 s) flagged: none; {'same_time_discordant': 0}
=== S1: 1050 participants; domains ['tremor', 'axial', 'brady', 'function', 'independence', 'cognition']
C12 DYNOTEARS  (0.1 s) flagged: ['function -> independence']; {'same_time_discordant': 0}
```

Under speed alone (S1), a link of the shared mechanism, function → independence, survives the threshold 0.1 in one phenotype's yearly graph and not in the other's. It is reported as a "difference". With equal speeds (S0) nothing is flagged in this dataset.

**Design choices and pitfalls.**
- Comparing thresholded supports makes the output depend on where yearly coefficients sit relative to $\tau_a$ and $\tau_w$; speed moves them across the threshold.
- DYNOTEARS assumes no hidden confounding and independent rows; neither holds for pooled patient transitions (protocol §7.2).

### 4.27 `c5_tsldiffpc` — C5: the difference graph between the phenotypes' two-slice tables

[`run.py:63`](../../src/pdx/comparators/run.py#L63)

**What it does.** Builds one window per phenotype, $Z_g=[\text{std}(x_t),\ \text{std}(x_{t+1})]$, runs the tsLDiffPC/tsDCI skeleton with separating sets (§4.43) and the orientation of the chosen variant (§4.45–§4.46), and reports:
- the retained lagged difference edges as a [to, from] matrix, own lags apart;
- the root causes (series heading a directed difference edge into the current block) and the root-cause pairs (members of undirected same-time difference edges).

**Inputs → outputs.** `panel`, `alpha` (0.05), `max_s` (3), `variant` (`"tsdci"` or `"tsldiffpc"`) → `cross_links`, `self_lags`, `root_causes`, `root_cause_pairs`.

**The mathematics.** A lagged window edge $(a,b)$ with $a<d\le b$ is the link "series $a$ at $t$ → series $b-d$ at $t+1$", so it goes to `lag[b - d, a]`. Regime 1 is TD ($Z_0$), regime 2 is PIGD ($Z_1$). The algorithms themselves are derived in §4.37–§4.47.

**From the formula to the code.** Lines 74–81 fill `lag` twice, from the directed set and from the skeleton's edges; every lagged edge is directed by time, so the second loop adds nothing new. Line 82 gets the root causes.

**Worked example** ([`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py)). The C5 lines:

```text
=== S0 ...
C5  tsDCI      (0.0 s) flagged: none; {'root_causes': [], 'root_cause_pairs': []}
C5  tsLDiffPC  (0.0 s) flagged: none; {'root_causes': [], 'root_cause_pairs': []}
=== S1 ...
C5  tsDCI      (0.0 s) flagged: ['own lag of axial']; {'root_causes': [1], 'root_cause_pairs': []}
C5  tsLDiffPC  (0.0 s) flagged: ['own lag of axial']; {'root_causes': [1], 'root_cause_pairs': []}
```

Under speed alone, axial's persistence differs on the yearly scale, so axial (series 1) is named a root cause: the signature the tsLDiffPC/tsDCI note found in about half of the domains (protocol Appendix E, check 5). The two variants share the skeleton and differ only in same-time orientation, so they agree here.

**Design choices and pitfalls.**
- `max_s` defaults to 3 at every $d$. Protocol §8.4 and the tsLDiffPC/tsDCI note use every conditioning set at six domains and a cap of 3 only at ten domains (Notes 4.N).
- Both variants are returned by separate calls; the benchmark must run each.

### 4.28 `_yearly` and `c3_nct` — C3: a permutation test on the yearly networks

[`run.py:89`](../../src/pdx/comparators/run.py#L89), [`run.py:95`](../../src/pdx/comparators/run.py#L95)

**What it does.** `_yearly` fits the yearly network by least squares: every domain at $t+1$ on all domains at $t$ with an intercept, returned [to, from]. `c3_nct` computes, from the two phenotypes' networks, a global strength difference, the largest cross-link difference and every edge's absolute difference, and refers each to its distribution under `n_perm` permutations of the participants' labels.

**Inputs → outputs.** `panel`, `n_perm` (2000), `rng`, `alpha` → `p_global_strength`, `p_max_edge`, `cross_links` and `self_lags` (edges rejected family-wise), `p_edges` (per-edge permutation p-values), `p_edges_familywise`.

**The mathematics.** With $\hat B_g$ the least-squares yearly matrix of phenotype $g$ ($[\hat B_g]_{jk}$: $x_{t,k}\to x_{t+1,j}$), and "off" the off-diagonal entries,

$$
T_{str}=\Big\lvert\sum_{\text{off}}\lvert\hat B_0\rvert-\sum_{\text{off}}\lvert\hat B_1\rvert\Big\rvert,\qquad
T_{max}=\max_{\text{off}}\lvert\hat B_0-\hat B_1\rvert,\qquad T_{jk}=\lvert\hat B_0-\hat B_1\rvert_{jk}\ \ (\text{all } d^2).
$$

Each permutation $\pi$ shuffles the participants' labels and maps them to their pairs. The p-values are $(\#\{T^*_\pi\ge T\}+1)/(n_{perm}+1)$ for $T=T_{str}$, $T_{max}$ and each $T_{jk}$. Per-edge family-wise decisions use a max-T on the edges scaled by their permutation SD:

$$
p^{fw}_{jk}=\frac{\#\big\{\pi:\ \max_{(a,b)}T^*_{\pi,ab}/\operatorname{sd}(T^*_{ab})\ \ge\ T_{jk}/\operatorname{sd}(T^*_{jk})\big\}+1}{n_{perm}+1},\qquad \text{reject if } p^{fw}_{jk}\le\alpha .
$$

**From the formulas to the code.** Lines 106–110 (`stats_for`) compute $T_{str}$, $T_{max}$ and the matrix of $T_{jk}$ for a label vector; line 116 `rng.permutation(groups)[who]` permutes by participant; lines 122–126 the per-edge and family-wise p-values (`sd` floored at $10^{-12}$).

**Worked example** ([`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py), 500 permutations). The C3 lines:

```text
=== S0 ...
C3  NCT-style  (0.1 s) flagged: none; {'p_global_strength': 0.138, 'p_max_edge': 0.643}
=== S1 ...
C3  NCT-style  (0.2 s) flagged: ['own lag of axial', 'own lag of independence']; {'p_global_strength': 0.042, 'p_max_edge': 0.397}
```

Under speed alone the global strength differs ($p=0.042$) and two persistences are flagged family-wise: speed read as a change in the network.

**Design choices and pitfalls.**
- The per-edge statistics are scaled but not centred, unlike `separate.py`. Both are valid max-T procedures (any per-edge transformation applied identically to the observed and permuted values keeps the family-wise error rate); they differ in power.
- The global and max-edge statistics leave out own lags; the per-edge family includes them.
- Permuting labels assumes the phenotypes are exchangeable under the null, which a speed difference violates; that is the point of a naive comparator.

### 4.29 `c11_naive` — C11: separate graphs read naively, and v0.3's interaction tests

[`run.py:133`](../../src/pdx/comparators/run.py#L133)

**What it does.** Two naive readings:
1. **Separate graphs, naively**: one fast Step 1 fit per phenotype on all its participants (no equal subsampling, no reference); the lagged adjacencies present in one graph and absent from the other.
2. **v0.3's link-wise interaction tests**: for each target $j$ and source $k$, a pooled regression of $x_{t+1,j}$ on all year-$t$ domains, the phenotype, and phenotype × $x_{t,k}$, with participant-clustered errors; Benjamini–Hochberg (BH) over all $d^2$ tests.

**Inputs → outputs.** `panel`, `cfg` (fast), `alpha` → `separate_graphs` and `interaction_tests` (each split by `_split`), and the matrix of interaction p-values.

**The mathematics.** For each $(j,k)$:

$$
x_{t+1,j}=a+\sum_{m=1}^{d}b_m x_{t,m}+\eta\,g+\delta\,g\,x_{t,k}+u,\qquad H_0:\delta=0 .
$$

With $\mathbf H$ the design (one row per pair) and $\hat u$ the residuals, $s_c=\sum_{r\in c}\mathbf H_r\hat u_r$ per participant, and

$$
\widehat V=(\mathbf H^\top\mathbf H)^{+}\Big(\sum_c s_cs_c^\top\Big)(\mathbf H^\top\mathbf H)^{+}\,\frac{N_c}{N_c-1},\qquad z=\hat\delta/\sqrt{\widehat V_{\delta\delta}},\qquad p=2\{1-\Phi(\lvert z\rvert)\}.
$$

BH then rejects every hypothesis whose p-value is at most $p_{(i^*)}$, where $i^*$ is the largest $i$ with $p_{(i)}\le i\alpha/d^2$ (`multiplicity.bh`). Under speed alone $\delta\neq0$ for most links, because the yearly coefficients themselves differ.

**From the formulas to the code.** Lines 139–143 the two Step 1 fits; lines 151–158 the regression, the sandwich (`np.add.at` sums the scores per participant; `pinv` for safety) and the z-test; line 159 BH.

**Worked example** ([`ex12_comparators.py`](../guide_examples/s04/ex12_comparators.py)). The C11 lines:

```text
=== S0 ...
C11 naive      (1.1 s) separate graphs: ['tremor -> function', 'cognition -> independence']
                       interaction tests: none
                       TD graph (800): ['tremor -> function', 'axial -> function', 'brady -> function', 'function -> independence', 'cognition -> independence', 'axial -> cognition']
                       PIGD graph (250): ['axial -> function', 'brady -> function', 'function -> independence', 'axial -> cognition']
=== S1 ...
C11 naive      (0.4 s) separate graphs: ['tremor -> function', 'axial -> function']
                       interaction tests: none
                       TD graph (800): ['axial -> function', 'brady -> function', 'function -> independence', 'cognition -> independence']
                       PIGD graph (250): ['tremor -> function', 'brady -> function', 'function -> independence', 'cognition -> independence']
```

(The last two lines of each block come from the example script, which also prints each phenotype's own graph.) Every link listed is a true link of the shared mechanism, so every "difference" is false:
- With equal speeds (S0), TD's 800 participants give FCI the power to find all six cross-links; PIGD's 250 miss tremor → function and cognition → independence.
- With PIGD twice as fast (S1), the disagreement runs both ways. PIGD's yearly effects are larger, so it finds the weak tremor → function link that TD misses, while it misses axial → function.
- The interaction tests flag nothing in either dataset.

**Design choices and pitfalls.**
- The two parts use different rows (§4.23).
- The interaction model frees one source at a time and keeps all other links shared, the "one at a time" pattern the protocol replaced by joint fits (§7.3.3).

### 4.30 `dynotears.py`, the module docstring — the DYNOTEARS model and objective

[`dynotears.py:1`](../../src/pdx/comparators/dynotears.py#L1)

**What it does.** States the model, the objective and the optimisation schedule of DYNOTEARS (Pamfil, Sriwattanaworachai, Desai et al. 2020), re-implemented from the authors' causalnex code (read, not run). The file is identical to the validated copy `scripts/toy/dynotears.py`, which reproduced the paper's published F1 scores to within 0.02 on average (largest gap 0.07; protocol §8.4).

**The mathematics.**

*The model.* One row per transition, with $x_t$ the current slice and $x_{t-1}$ the lagged one (one lag here):

$$
x_t^\top=x_t^\top W+x_{t-1}^\top A+z_t^\top,\qquad\text{stacked: }\ \mathbf X=\mathbf XW+\mathbf YA+\mathbf Z ,
$$

where $W_{ab}$ is a same-time effect $a\to b$ and $A_{ab}$ a lagged effect $a(t-1)\to b(t)$, both [from, to]. In the code, `X` is the current slice and `Y` the lagged slice.

*The objective.*

$$
\min_{W,A}\ \underbrace{\frac1{2n}\big\lVert\mathbf X(I-W)-\mathbf YA\big\rVert_F^2}_{\ell(W,A)}+\lambda_w\lVert W\rVert_1+\lambda_a\lVert A\rVert_1\qquad\text{subject to}\qquad h(W)=\operatorname{tr}\big(e^{W\circ W}\big)-d=0 .
$$

*Why $h(W)=0$ means acyclic* (Zheng et al. 2018). With $M=W\circ W$ (elementwise squares, so $M\ge0$),

$$
h(W)=\operatorname{tr}\Big(\sum_{k\ge0}\frac{M^k}{k!}\Big)-d=\sum_{k\ge1}\frac{\operatorname{tr}(M^k)}{k!}.
$$

$\operatorname{tr}(M^k)$ is the sum, over all closed walks of length $k$, of the products of the squared weights along the walk. Every term is non-negative, so $h(W)=0$ exactly when there is no closed walk: no cycle. Its gradient is $\nabla h(W)=(e^{W\circ W})^\top\circ2W$. Only $W$ needs the constraint: lagged effects point forward in time and cannot form cycles.

*The augmented Lagrangian.* The constrained problem is solved through

$$
L_\rho(W,A;\alpha)=\ell(W,A)+\lambda_w\lVert W\rVert_1+\lambda_a\lVert A\rVert_1+\frac\rho2h(W)^2+\alpha\,h(W),
$$

minimised over $(W,A)$ for fixed $(\rho,\alpha)$. After each minimisation: if $h$ did not fall below a quarter of its previous value, $\rho\leftarrow10\rho$ and the minimisation is repeated; otherwise the dual variable moves, $\alpha\leftarrow\alpha+\rho h$. The loop stops when $h\le10^{-8}$ or after 100 dual steps. Large $\rho$ forces $h$ towards 0.

*The L1 terms.* Writing $W=W^+-W^-$ and $A=A^+-A^-$ with all parts $\ge0$ makes $\lVert W\rVert_1=\mathbf 1^\top(W^++W^-)$ linear (at the optimum at most one part of each entry is positive). The problem becomes smooth with box constraints, which L-BFGS-B handles. The bounds also fix $W_{aa}=0$: no self-loops in the same slice.

*Thresholds.* At the end, $\lvert W_{ab}\rvert<\tau_w$ and $\lvert A_{ab}\rvert<\tau_a$ are set to zero. The authors' code applies one threshold to both; the paper's simulations use separate ones, so both are arguments.

**Inputs → outputs.** None (a docstring); the fit is `dynotears` (§4.35).

**Worked example.** §4.35 fits a three-variable system with a known graph.

**Design choices and pitfalls.** The loss is computed from cross-product matrices (§4.33), which is algebraically the authors' residual form; the example in §4.35 checks the equality numerically.

### 4.31 `_layout` — the size of the parameter vector and its bounds

[`dynotears.py:25`](../../src/pdx/comparators/dynotears.py#L25)

**What it does.** Returns the number of same-time entries ($d^2$), of lagged entries ($pd^2$), and the L-BFGS-B bounds for the vector $[W^+, W^-, A^+_1, A^-_1, \dots]$.

**Inputs → outputs.** $d$, $p$ (lags) → `nw`, `na`, a list of $2(d^2+pd^2)$ bounds.

**The mathematics.** $W^\pm_{ab}\in[0,\infty)$ for $a\ne b$ and $W^\pm_{aa}\in[0,0]$; $A^\pm_{ab}\in[0,\infty)$ for all $a,b$, own lags included.

**Worked example.** In §4.35 the estimated $W$ has an exact zero diagonal, and $\hat A$ has non-zero own lags.

**Design choices and pitfalls.** `2 * bw` repeats the $W$ bounds for $W^+$ and $W^-$; the authors' optional "tabu" edges are not implemented.

### 4.32 `_unpack` — from the vector to $(W, A)$

[`dynotears.py:31`](../../src/pdx/comparators/dynotears.py#L31)

**What it does.** Rebuilds $W=W^+-W^-$ ($d\times d$) and $A=A^+-A^-$ ($pd\times d$, lag 1 first) from the optimisation vector, in the authors' `_reshape_wa` layout.

**Inputs → outputs.** `v`, $d$, $p$ → $(W, A)$.

**The mathematics.** The vector is $[\operatorname{vec}W^+,\ \operatorname{vec}W^-,\ \operatorname{vec}A^+_1,\ \operatorname{vec}A^-_1,\ \dots,\ \operatorname{vec}A^+_p,\ \operatorname{vec}A^-_p]$ (row-major). The lagged part reshaped to $2p$ rows of $d^2$ has $A^+_\ell$ in the even rows and $A^-_\ell$ in the odd rows, hence `a[::2] - a[1::2]`.

**Worked example.** §4.35 uses `_unpack` to evaluate the loss in residual form.

**Design choices and pitfalls.** The gradient in `_objective` must be packed in the same order; `gradient_check` (§4.36) is what confirms it.

### 4.33 `_objective` — the augmented Lagrangian and its gradient

[`dynotears.py:40`](../../src/pdx/comparators/dynotears.py#L40)

**What it does.** Returns $L_\rho$ and its gradient with respect to the split vector, for given cross-products, penalties, $\rho$ and $\alpha$.

**Inputs → outputs.** `v`, `S = (Sxx, Sxy, Syy)`, $d$, $p$, $\lambda_w$, $\lambda_a$, $\rho$, $\alpha$ → $(f, g)$.

**The mathematics.** With $S_{xx}=\mathbf X^\top\mathbf X/n$, $S_{xy}=\mathbf X^\top\mathbf Y/n$, $S_{yy}=\mathbf Y^\top\mathbf Y/n$ and $M=I-W$, expanding the squared Frobenius norm gives

$$
\ell=\tfrac12\Big[\operatorname{tr}(M^\top S_{xx}M)-2\operatorname{tr}(M^\top S_{xy}A)+\operatorname{tr}(A^\top S_{yy}A)\Big].
$$

Differentiating (using $\partial\operatorname{tr}(M^\top SM)/\partial M=2SM$ for symmetric $S$, and $\partial M/\partial W=-I$):

$$
\frac{\partial\ell}{\partial W}=-(S_{xx}M-S_{xy}A),\qquad \frac{\partial\ell}{\partial A}=-(S_{xy}^\top M-S_{yy}A),\qquad
\frac{\partial}{\partial W}\Big(\frac\rho2h^2+\alpha h\Big)=(\rho h+\alpha)\,(e^{W\circ W})^\top\circ2W .
$$

For the split variables, $\partial f/\partial W^+=g_W+\lambda_w$ and $\partial f/\partial W^-=-g_W+\lambda_w$, and likewise for $A$.

**From the formulas to the code.**

| line | code | formula |
|---|---|---|
| 44–46 | `M = I - W`; `E = expm(W * W)`; `h = trace(E) - d` | $M$, $e^{W\circ W}$, $h$ |
| 47 | `np.sum(M * (Sxx @ M))` etc. | $\operatorname{tr}(M^\top S_{xx}M)$: $\sum_{ab}M_{ab}(S_{xx}M)_{ab}$ |
| 48 | `loss + 0.5*rho*h*h + alpha*h + lambda_w*sum(v_W) + lambda_a*sum(v_A)` | $L_\rho$ (the L1 terms are sums of the split parts) |
| 49 | `-(Sxx @ M - Sxy @ A) + (rho*h + alpha) * E.T * W * 2` | $g_W$ |
| 50 | `-(Sxy.T @ M - Syy @ A)` reshaped to $p$ rows of $d^2$ | $g_A$, lag by lag |
| 51–53 | `[gW, -gW, (gA, -gA) per lag] + lambda` | the split gradient in `_unpack`'s order |

**Worked example.** §4.35: the loss from cross-products equals the residual form to ten decimals, and the gradient matches central differences to $10^{-9}$.

**Design choices and pitfalls.** The cross-products are computed once per fit, so each evaluation costs $O(d^3)$ instead of $O(nd^2)$; with thousands of transitions this is the main saving over the authors' code.

### 4.34 `_h` — the acyclicity function alone

[`dynotears.py:57`](../../src/pdx/comparators/dynotears.py#L57)

**What it does.** Returns $h(W)=\operatorname{tr}(e^{W\circ W})-d$ for a vector, to judge progress between inner solves.

**Inputs → outputs.** `v`, $d$, $p$ → a non-negative number (up to rounding).

**The mathematics.** §4.30.

**Worked example.** §4.35 prints $h=7.1\times10^{-9}$ at the end of the fit.

**Design choices and pitfalls.** None.

### 4.35 `dynotears` — the fit

[`dynotears.py:62`](../../src/pdx/comparators/dynotears.py#L62)

**What it does.** Runs the augmented-Lagrangian schedule from zero, then thresholds.

**Inputs → outputs.** `X` ($n\times d$, current slice), `Y` ($n\times pd$, lagged slices, lag 1 first), $\lambda_w$, $\lambda_a$, $\tau_w$, $\tau_a$, `max_iter` (100), `h_tol` ($10^{-8}$) → $(W, A, \text{info})$ with info = final $h$ (before thresholding), final $\rho$, and whether $h\le$ `h_tol`.

**From the method to the code.**

| line | code | meaning |
|---|---|---|
| 66 | `S = (X.T @ X / n, X.T @ Y / n, Y.T @ Y / n)` | cross-products, once |
| 68–70 | `v_est = 0`; `rho, alpha, h_val, h_new = 1, 0, inf, inf` | the authors' start |
| 72 | `for _ in range(max_iter)` | dual steps |
| 73 | `while rho < 1e20 and (h_new > 0.25 * h_val or h_new == inf)` | re-solve with larger $\rho$ until $h$ falls by a factor 4 (in the first step $h_{val}=\infty$, so one solve) |
| 74–75 | `sopt.minimize(..., v_est, method="L-BFGS-B", jac=True, bounds=bnds)` | the inner problem, always started from the last accepted estimate |
| 77–78 | `rho *= 10` | the penalty step |
| 79–81 | `v_est = v_new`; `h_val = h_new`; `alpha += rho * h_val` | accept; dual ascent |
| 82–84 | `if h_val <= h_tol` | converged |
| 85–87 | `_unpack`; zero the small entries | thresholds |

**Worked example** ([`ex13_dynotears.py`](../guide_examples/s04/ex13_dynotears.py)). Three variables, a same-time chain $x_0\to x_1\to x_2$ (weights 0.8), own lags 0.5 and one cross-lag $x_0(t-1)\to x_2(t)$ (0.4); 2,000 transitions.

```python
W[0, 1] = W[1, 2] = 0.8; A = 0.5 * np.eye(3); A[0, 2] = 0.4
x[t] = (x[t - 1] @ A + rng.normal(size=d)) @ np.linalg.inv(np.eye(d) - W)   # solves x_t = x_t W + x_{t-1} A + z_t
W1, A1, info1 = DY.dynotears(X, Y, 0.05, 0.05, tau_w=0.3, tau_a=0.1)
```

```text
0.09 s for both fits; info: {'h': 7.109926425385993e-09, 'rho': 1000000000000.0, 'converged': True}
W estimated, no threshold:
 [[0.    0.756 0.038]
 [0.    0.    0.804]
 [0.    0.    0.   ]]
W estimated, tau_w = 0.3 (truth: 0.8 at [0, 1] and [1, 2]):
 [[0.    0.756 0.   ]
 [0.    0.    0.804]
 [0.    0.    0.   ]]
A estimated, tau_a = 0.1 (truth: 0.5 on the diagonal, 0.4 at [0, 2]):
 [[0.476 0.    0.322]
 [0.    0.478 0.   ]
 [0.    0.    0.503]]
h(W) of the final estimate: 7.11e-09
loss from cross-products 16.4078206224; residual form 16.4078206224
gradient check (largest relative error against central differences): 4.5e-10
```

Reading the output:
- The structure is recovered exactly after thresholding: the small same-time entry $x_0\to x_2$ (0.038) falls below $\tau_w=0.3$.
- The weights are shrunk by the L1 penalty: 0.756 for 0.8, and 0.322 for the cross-lag 0.4.
- $h$ reached $7\times10^{-9}$ with $\rho=10^{12}$: the schedule needed twelve tenfold increases of the penalty.
- The cross-product loss equals the residual form, and the analytic gradient matches central differences (§4.36).

**Design choices and pitfalls.**
- The search starts at zero, as in the authors' code; DYNOTEARS' problem is non-convex, so other starts could give other graphs.
- `converged` is False if 100 dual steps or $\rho\ge10^{20}$ end the loop with $h>10^{-8}$; `c12_dynotears` does not check it.

### 4.36 `gradient_check` — the analytic gradient against central differences

[`dynotears.py:91`](../../src/pdx/comparators/dynotears.py#L91)

**What it does.** On random data and a random positive point, compares `_objective`'s gradient with central finite differences and returns the largest absolute difference divided by the largest gradient component.

**Inputs → outputs.** `seed`, $d$, $p$, $n$ → a relative error (about $10^{-9}$ to $10^{-10}$ when correct).

**The mathematics.** $\partial f/\partial v_k\approx\{f(v+he_k)-f(v-he_k)\}/2h$ with $h=10^{-6}$, error $O(h^2)$.

**Worked example.** §4.35, last line: $4.5\times10^{-10}$. The validation file `scripts/toy/toy_dynotears_validation_results.json` records $1.3\times10^{-9}$.

**Design choices and pitfalls.** No unit test calls it; it is run by the toy validation script and by this guide's example.

### 4.37 `ts_ldiffpc.py`, the module docstring — difference graphs for time series

[`ts_ldiffpc.py:1`](../../src/pdx/comparators/ts_ldiffpc.py#L1)

**What it does.** States what is re-implemented from the authors' pyCIPHOD code (Ruer, Loranchet, Bystrova & Assaad 2026; read, not run) and how the skeleton works. The file is identical to `scripts/toy/ts_ldiffpc.py`, the version that reproduced the paper's Figure 4 within Monte Carlo error (`docs/tsldiffpc_tsdci_investigation.md`, §3).

**Inputs → outputs.** None (a docstring); the functions follow in §4.38–§4.47.

**Worked example.** [`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py) runs the algorithms on three two-regime systems with known differences (§4.38–§4.47).

**The mathematics.**

*Setting.* Two regimes $r=1,2$ (here TD and PIGD), each a linear model with lagged and same-time effects:

$$
x_j(t)=\sum_k a^{(r)}_{jk}\,x_k(t-1)+\sum_{m\ne j}c^{(r)}_{jm}\,x_m(t)+\varepsilon^{(r)}_j(t).
$$

The *difference graph* has a lagged edge $x_k(t-1)\to x_j(t)$ exactly when $a^{(1)}_{jk}\ne a^{(2)}_{jk}$, and a same-time edge when $c^{(1)}_{jm}\ne c^{(2)}_{jm}$. A series with a changed incoming coefficient, its own lag included, is an "effect-defying root cause". The methods look for these edges directly, without learning either regime's graph.

*The window.* Columns $0..d-1$ are $x(t-1)$, columns $d..2d-1$ are $x(t)$. Candidate edges: every lag → current pair ($d^2$, own lags included) and every current–current pair ($d(d-1)/2$). Lag–lag pairs are never candidates.

*The removal rule.* For a candidate $(x,y)$ and a conditioning set $\mathbf S$ of other window nodes (lag *and* current, sizes $0..$`max_s`), regress $y$ on $[1,x,\mathbf S]$ in each regime and test equality of the coefficient of $x$ (§4.39). The edge is removed at the **first** $(\mathbf S,\text{direction})$ with $p>\alpha$, in either direction ($y$ on $x$, or $x$ on $y$). "Diff-adjacency faithfulness" (their A5) assumes a changed coefficient differs for every set, so in theory nothing true is removed; in finite samples, a true difference survives only if *every* test rejects. That is why the search is lenient: it rarely invents edges, and it misses many.

The number of tests for an edge that survives is $2\sum_{s=0}^{s_{max}}\binom{2d-2}{s}$: 2,048 at six domains without a cap, 352 at six domains with a cap of 3, and 1,976 at ten domains with a cap of 3.

*One skeleton for both.* The admissible sets do not depend on the current graph, so each candidate is decided on its own and tsLDiffPC and tsDCI return the same skeleton. They differ in how same-time edges are oriented (§4.45, §4.46). Lagged edges are oriented by time in both.

**Design choices and pitfalls.** The tests treat rows as independent and use classical standard errors; there is no multiplicity correction. In this study the regimes are phenotypes with different speeds, so the coefficients differ on the yearly scale, persistence most (protocol Appendix E, check 5).

### 4.38 `gram` and `coef_se` — least squares from a Gram matrix

[`ts_ldiffpc.py:22`](../../src/pdx/comparators/ts_ldiffpc.py#L22), [`ts_ldiffpc.py:27`](../../src/pdx/comparators/ts_ldiffpc.py#L27)

**What it does.** `gram` returns $\tilde Z^\top\tilde Z$ for $\tilde Z=[\mathbf 1, Z]$, and $n$. `coef_se` returns the least-squares coefficient of column $x$ in the regression of column $y$ on $[1,x,\mathbf S]$, and its classical standard error, using only the Gram matrix.

**Inputs → outputs.** `gram`: $Z$ → $(G, n)$. `coef_se`: $G$, $n$, $y$, $x$, $\mathbf S$ (indices into $Z$) → $(\hat b_x,\ \widehat{\mathrm{se}})$. (In §4.38–§4.44, $G$ is this Gram matrix, as in the code, not the endpoint matrix $\mathcal G$.)

**The mathematics.** With the index set $I=\{0,\ 1+x,\ 1+\mathbf S\}$ (0 is the intercept),

$$
\hat\beta=G_{II}^{-1}G_{I,1+y},\qquad \mathrm{RSS}=G_{1+y,1+y}-\hat\beta^\top G_{I,1+y},\qquad
\hat\sigma^2=\frac{\mathrm{RSS}}{n-\lvert I\rvert},\qquad \widehat{\mathrm{se}}(\hat b_x)=\sqrt{\hat\sigma^2\,[G_{II}^{-1}]_{11}} .
$$

The residual sum of squares follows from the normal equations: $\lVert y-X\hat\beta\rVert^2=y^\top y-2\hat\beta^\top X^\top y+\hat\beta^\top X^\top X\hat\beta=y^\top y-\hat\beta^\top X^\top y$.

**From the formulas to the code.** Line 29 builds $I$; lines 30–31 take $G_{II}$ and $G_{I,1+y}$; lines 32–35 invert (pseudo-inverse if singular); line 36 $\hat\beta$; line 37 $\hat\sigma^2$; line 38 returns $\hat\beta_1$ (position 1 of $I$ is $x$) and the standard error, with negative rounding clipped to 0.

**Worked example** ([`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py)). Three series in two regimes (2,000 transitions each); regime 2 adds the link $x_0(t-1)\to x_2(t)$. The regression of $x_2(t)$ on $[1, x_0(t-1), x_1(t)]$ in regime 1, against statsmodels:

```python
G1, G2 = TS.gram(Z1), TS.gram(Z2)
b, se = TS.coef_se(*G1, 5, 0, [4])                    # window columns: 5 = x2(t), 0 = x0(t-1), 4 = x1(t)
ols = sm.OLS(Z1[:, 5], sm.add_constant(Z1[:, [0, 4]])).fit()
```

```text
coef_se: b = 0.003894, se = 0.026592;  statsmodels: b = 0.003894, se = 0.026592
```

**Design choices and pitfalls.** One Gram matrix per regime serves every test, which is what makes the re-implementation fast (a fraction of a second per fit, against data-frame refits in pyCIPHOD).

### 4.39 `wald_p` — the two-sample test of equal coefficients

[`ts_ldiffpc.py:41`](../../src/pdx/comparators/ts_ldiffpc.py#L41)

**What it does.** Fits the same regression in both regimes and tests whether the coefficient of $x$ is the same.

**Inputs → outputs.** The two Gram tuples, $y$, $x$, $\mathbf S$ → a p-value; 1.0 if both standard errors are zero; NaN if a fit fails.

**The mathematics.** The regimes are independent samples, so under $H_0:b^{(1)}=b^{(2)}$

$$
z=\frac{\hat b^{(1)}-\hat b^{(2)}}{\sqrt{\widehat{\mathrm{se}}_1^2+\widehat{\mathrm{se}}_2^2}}\ \approx\ N(0,1),\qquad p=2\{1-\Phi(\lvert z\rvert)\}.
$$

**Worked example** (same script):

```text
wald_p: 1.91e-41;  by hand: 1.91e-41  (b1 = 0.004, b2 = 0.512)
```

The new lagged link $x_0(t-1)\to x_2(t)$ of regime 2 gives coefficients 0.004 and 0.512, a very clear difference.

**Design choices and pitfalls.** A NaN p-value never removes an edge (`NaN > alpha` is False), as pyCIPHOD skips failed tests. In practice `coef_se` falls back to a pseudo-inverse, so NaN is rare.

### 4.40 `skeleton` — the shared skeleton, candidate by candidate

[`ts_ldiffpc.py:52`](../../src/pdx/comparators/ts_ldiffpc.py#L52)

**What it does.** For every candidate edge, tries conditioning sets of increasing size from all other window nodes, in both regression directions, and removes the edge at the first non-significant test.

**Inputs → outputs.** $Z^{(1)}$, $Z^{(2)}$, $d$ (`p` in the code), `alpha`, `max_s` (default $2d-2$, all) → `{(x, y): retained}` for every candidate, and the number of tests run.

**The mathematics.** With $V$ the $2d$ window nodes: retained$(x,y)$ $\iff$ $p\le\alpha$ for every $\mathbf S\subseteq V\setminus\{x,y\}$ with $\lvert\mathbf S\rvert\le s_{max}$ and for both directions.

**Worked example** ([`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py), case a). Three series; regime 2 adds $x_0(t-1)\to x_2(t)$ with coefficient 0.4.

```python
B2[2, 0] = 0.4                                         # (a) new link x0(t-1) -> x2(t) in regime 2
kept, ntest = TS.skeleton(Z1, Z2, p)
```

```text
(a) skeleton: 70 tests, 0.00 s
    retained lagged difference edges F[to, from]:
 [[0 0 0]
 [0 0 0]
 [1 0 0]]
    retained same-time pairs: []
    tsdci    : directed [(0, 5)], undirected [], root causes ({2}, set())
    tsldiffpc: directed [(0, 5)], undirected [], root causes ({2}, set())
```

The one retained edge is the true difference. It needed all $2\times2^4=32$ tests to reject; the other eleven candidates were removed after the remaining 38 tests. Both variants direct it by time, window node 0 → 5, so series 2 is the root cause (the last two lines come from `skeleton_with_sepsets`, `orient` and `root_causes`, §4.43–§4.47).

**Design choices and pitfalls.** The function is used by the toy scripts; `c5_tsldiffpc` uses `skeleton_with_sepsets`, which gives the same edges and also records the separating sets.

### 4.41 `lag_matrix` and `contemporaneous_pairs` — reading the skeleton

[`ts_ldiffpc.py:78`](../../src/pdx/comparators/ts_ldiffpc.py#L78), [`ts_ldiffpc.py:87`](../../src/pdx/comparators/ts_ldiffpc.py#L87)

**What it does.** `lag_matrix` returns $F_{jk}=$ retained$(k,\ d+j)$, the lagged difference edges [to, from]. `contemporaneous_pairs` lists the retained same-time pairs $(j,m)$, $j<m$.

**Inputs → outputs.** `kept`, $d$ → a boolean $d\times d$ matrix; a list of pairs.

**The mathematics.** None.

**Worked example.** §4.40: $F_{20}=1$ is "$x_0$ → $x_2$".

**Design choices and pitfalls.** None.

### 4.42 `_order` — pyCIPHOD's node order

[`ts_ldiffpc.py:96`](../../src/pdx/comparators/ts_ldiffpc.py#L96)

**What it does.** Sorts the window nodes as pyCIPHOD sorts its time-variable objects: by series, then the lag before the current value: $x_0(t-1), x_0(t), x_1(t-1), x_1(t),\dots$, that is, indices $0, d, 1, d+1, \dots$.

**Inputs → outputs.** $d$ → a list of $2d$ indices.

**The mathematics.** None. The order matters only through the loops: which separating set is found first, which edges are visited first in orientation.

**Worked example** ([`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py)):

```text
pyCIPHOD's node order for d = 3: [0, 3, 1, 4, 2, 5]
```

**Design choices and pitfalls.** Reproducing the loop order is what makes the re-implementation give pyCIPHOD's results, not just results from the same algorithm.

### 4.43 `skeleton_with_sepsets` — the skeleton in each algorithm's loop order, with separating sets

[`ts_ldiffpc.py:101`](../../src/pdx/comparators/ts_ldiffpc.py#L101)

**What it does.** The same removal rule as `skeleton`, with the loop order of each algorithm, recording the set that removed each edge:
- **tsDCI:** candidate by candidate (in pyCIPHOD's order), sets by size, both directions for each set;
- **tsLDiffPC:** level by level ($s=0,1,\dots$); at each level a snapshot of every node's neighbours; each node visits its neighbours and tests one direction ($y$ on $x$ for the visit from $x$). Both endpoints visit, so both directions are tried.

**Inputs → outputs.** $Z^{(1)}$, $Z^{(2)}$, $d$, `alpha`, `max_s`, `variant` → `edges` (a set of `frozenset` pairs) and `sepset` (`{(a, b): S}`, both orders).

**The mathematics.** As in §4.40; the separating sets feed tsLDiffPC's collider rule (§4.46).

**Worked example** ([`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py), case c). A same-time link $x_1(t)\to x_2(t)$ that is 0.5 in regime 1 and 1.0 in regime 2:

```text
(c) a stronger same-time link x1(t) -> x2(t): 0.5 in regime 1, 1.0 in regime 2 (window nodes 4 -> 5):
    tsdci    : edges [], directed [], undirected [], (heads, pairs) (set(), set())
    tsldiffpc: edges [], directed [], undirected [], (heads, pairs) (set(), set())
    separating set recorded for (4, 5): [2]
    with S = [2]: x2(t) on x1(t): b1 = 0.615, b2 = 1.072, p = 1.8e-61;  x1(t) on x2(t): b1 = 0.542, b2 = 0.560, p = 0.37
```

The true difference is missed by both variants. The set printed is the one the tsLDiffPC loop recorded, $\{x_2(t-1)\}$ (window node 2). In the natural direction ($x_2$ on $x_1$) the coefficients differ hugely. But in the reverse regression, given $x_2(t-1)$, they happen to be nearly equal (0.542 and 0.560), and one non-significant test is enough to remove the edge. This is the leniency of §4.37 in its simplest form.

**Design choices and pitfalls.** In the tsLDiffPC loop, an edge removed from one endpoint is visited again from the other endpoint in the same level (the snapshot still lists it), and a second removal overwrites the recorded set. This mirrors the docstring's reading of pyCIPHOD; it can change which set is recorded, and so the collider decisions.

### 4.44 `resid_var` — a residual variance from the Gram matrix

[`ts_ldiffpc.py:149`](../../src/pdx/comparators/ts_ldiffpc.py#L149)

**What it does.** The residual variance of column $y$ after least squares on $[1,\mathbf S]$, and its degrees of freedom.

**Inputs → outputs.** $G$, $n$, $y$, $\mathbf S$ → $(\hat\sigma^2,\ n-\lvert I\rvert)$ with $I=\{0\}\cup(1+\mathbf S)$.

**The mathematics.** As in §4.38, with a pseudo-inverse as in pyCIPHOD's helper.

**Worked example.** Used inside §4.45.

**Design choices and pitfalls.** None.

### 4.45 `orient`, the tsDCI branch — same-time edges by invariance of residual variances

[`ts_ldiffpc.py:157`](../../src/pdx/comparators/ts_ldiffpc.py#L157) (the tsDCI part is lines 168–205)

**What it does.** Lagged difference edges are directed by time. For each same-time difference edge $i$–$j$ (in pyCIPHOD's order), it walks through pairs of conditioning sets $(m_i, m_j)$, the power sets of all other window nodes taken in parallel, and stops at the first set for which one endpoint's residual variance is the same in both regimes.

**Inputs → outputs.** `edges`, `sepset`, $Z^{(1)}$, $Z^{(2)}$, $d$, `alpha`, `variant="tsdci"` → `directed` (a set of (tail, head)) and `undirected`.

**The mathematics.** *The F-test.* With Gaussian errors and equal residual variances,

$$
F=\hat\sigma^2_{(1)}\big/\hat\sigma^2_{(2)}\ \sim\ F(\nu_1,\nu_2),\qquad p=2\min\{F_{\nu_1,\nu_2}(F),\ 1-F_{\nu_1,\nu_2}(F)\}.
$$

*The orientation rule* (DCI, Wang 2018, carried to time series). If the noise variances are equal in both regimes, the residual variance of a node given its parents is the same in both, whatever its coefficients do; given a set that contains one of its children whose mechanism changed, it is not. So, for the first set $m_i$ with an invariant residual variance of $i$:

$$
j\in m_i\ \Rightarrow\ j\to i,\qquad j\notin m_i\ \Rightarrow\ i\to j,
$$

and symmetrically for $j$ with $m_j$. The equal-noise premise is exactly what a speed difference breaks: speed changes the yearly noise.

**From the method to the code.** Lines 162–167 split lagged and same-time edges. `var_p` (171–180) is the F-test, returning None when a fit fails or a variance is zero. Lines 186–204: for each undirected edge, `zip(powerset(not_i), powerset(not_j))`; the first invariant set orients the edge and the loop breaks.

**Worked example** ([`ex14_tsldiffpc.py`](../guide_examples/s04/ex14_tsldiffpc.py), case b). A same-time link $x_1(t)\to x_2(t)$ that exists only in regime 2 (0.8):

```text
(b) a new same-time link x1(t) -> x2(t): 0 in regime 1, 0.8 in regime 2 (window nodes 4 -> 5):
    tsdci    : edges [[4, 5]], directed [(4, 5)], undirected [], (heads, pairs) ({2}, set())
    tsldiffpc: edges [[4, 5]], directed [], undirected [[4, 5]], (heads, pairs) (set(), {1, 2})
    separating set recorded for (4, 5): none (edge kept)
    with S = []: x2(t) on x1(t): b1 = 0.224, b2 = 1.234, p = 1.7e-167;  x1(t) on x2(t): b1 = 0.196, b2 = 0.397, p = 2.1e-19
```

tsDCI orients the edge correctly as $x_1(t)\to x_2(t)$ (window nodes 4 → 5), so series 2 is the root cause. tsLDiffPC cannot orient a lone same-time edge (no collider), so it returns the pair {1, 2}.

**Design choices and pitfalls.**
- The power sets have $2^{2d-1}$ members. At ten domains that is 524,288 pairs of F-tests per edge in the worst case; the loop stops at the first invariant set, which usually comes early.
- The sets range over all window nodes, so a lagged node can be in $m_i$ and decide nothing about $j$; only membership of $j$ matters.

### 4.46 `orient`, the tsLDiffPC branch — colliders and time-restricted Meek rules

[`ts_ldiffpc.py:207`](../../src/pdx/comparators/ts_ldiffpc.py#L207)

**What it does.** After the lagged edges are directed by time:
1. **Unshielded colliders into the current block**: for $x$–$y$–$z$ with $y$ and $z$ current, $x$ and $z$ not adjacent, and $y\notin\operatorname{Sepset}(x,z)$, orient $x\to y\leftarrow z$.
2. **Meek's rules, restricted to time** (an edge is oriented only within one block), until nothing changes:
   - R1: $x\to y$, $y$–$z$, $x,z$ not adjacent ⇒ $y\to z$;
   - R2: $x\to y\to z$ and $x$–$z$ ⇒ $x\to z$;
   - R3: $x\to y\leftarrow z$ with $x,z$ not adjacent, and $w$–$x$, $w$–$y$, $w$–$z$ ⇒ $w\to y$;
   - R4: $x$–$y$, $x\to w$, $y$–$w$, $w\to z$, $y$–$z$ ⇒ $y\to z$.

**Inputs → outputs.** As in §4.45, with `variant="tsldiffpc"` (the data are not used in this branch).

**The mathematics.** The collider rule is PC's: a non-adjacent pair separated by a set without $y$ cannot have $y$ as a chain or fork between them. Meek's rules (Meek 1995) propagate orientations without creating new colliders or cycles. Meek's R4 in its standard form also requires $x$ and $z$ to be **non-adjacent**; the code does not check this (Notes 4.N).

**Worked example.** §4.45, case b: the lone same-time edge stays undirected in tsLDiffPC.

**Design choices and pitfalls.**
- When two colliders disagree about one edge, `set_dir` lets the last one win.
- Undirected-edge membership is symmetric (`frozenset`), the paper's intended meaning; the investigation note records that pyCIPHOD's own storage could not be checked.

### 4.47 `root_causes` — the certain root causes and the pairs

[`ts_ldiffpc.py:269`](../../src/pdx/comparators/ts_ldiffpc.py#L269)

**What it does.** Returns $\hat C$, the series heading a directed difference edge into the current block, and $\hat C_P$, the series in undirected same-time difference edges (each such pair contains at least one root cause).

**Inputs → outputs.** `directed`, `undirected`, $d$ → two sets of series indices.

**The mathematics.** $\hat C=\{b \bmod d:\ (a,b)\in\text{directed},\ b\ge d\}$, $\hat C_P=\{v \bmod d:\ v\in e,\ e\in\text{undirected}\}$.

**Worked example.** §4.40: $\hat C=\{2\}$ (the head of the new lagged link); §4.45: $\{2\}$ for tsDCI, $\hat C_P=\{1,2\}$ for tsLDiffPC.

**Design choices and pitfalls.** An own-lag difference makes its series a root cause, which is how speed produces false root causes (§4.27).

### 4.N Notes for the author

Things noticed while reading and running the code of this chapter. Nothing in `src/`, `tests/`, `scripts/` or `configs/` was changed. The reproductions are in `docs/guide_examples/s04/` and use synthetic data only.

**Can change a reported mark or a reproducibility claim**

1. **False tails survive the replacement of `fci_orient_bk`.** [`step1.py:155–162`](../../src/pdx/step1.py#L155) swaps causal-learn's `fci_orient_bk`, but its `is_arrow_point_allowed` (`causallearn/search/ConstraintBased/FCI.py:294–305`) still refuses an arrowhead at $y$ on $x\ {*\!\!-\!\!*}\ y$ whenever $x\to y$ is forbidden and $x$ already has an arrowhead. After `_orient_tiers` every later-tier node has one, so:
   - collider orientation (R0, `FCI.py:332–335`), R2, R3 and R4's collider branch can never put an arrowhead at the earlier end of a cross-tier edge, and `<->` cannot appear between tiers;
   - when a hidden cause meets a collider at a year-$t$ node, R0 skips the collider and R1 (`FCI.py:360–378`) then turns the remaining circle into a tail.

   Reproduction: [`ex04_orient_tiers.py`](../guide_examples/s04/ex04_orient_tiers.py), case 4. The truth $B\to A_t\leftarrow L\to C_{t+1}$ ($L$ hidden) gives `B o-> A_t; A_t --> C_t1`; the correct PAG, `A_t <-> C_t1`, comes back when the arrowhead check looks only at the existing mark. A forbidden direction never rules out an arrowhead in PAG semantics; only an existing tail (or a *required* edge) does. Adjacencies are unaffected, so the support of H-P is not. Affected: `lagged_directed` and so `D_directed` in `separate.py`, the marks of `reported_map`, and protocol §6.3's rule that an entry is read causally only where Step 1 orients it. Consider swapping `FCI.is_arrow_point_allowed` during the call as well, and adding case 4 as a unit test (`test_hidden_confounding_keeps_a_circle_at_the_earlier_node` uses two variables, which cannot form a collider).

2. **FCI's output depends on Python's string-hash seed.** causal-learn's `getPossibleDsep` (`FCI.py:188–266`) starts its search from a Python `set` of nodes (`adjacentNodes = set(...)`, line 199) and keeps `set`s of predecessors; nodes hash by name, so the order of the search, and with it the Possible-D-SEP sets, can differ from process to process. Reproduction: [`ex16_hash_seed.py`](../guide_examples/s04/ex16_hash_seed.py) runs one fit (scenario S1, six domains, clustered test, cap 4) in two fresh processes:

   ```text
   same lagged adjacency: True
   edges only with PYTHONHASHSEED=0: ['y2_t o-o y3_t', 'y3_t o-> y3_t1', 'y3_t o-o y4_t', 'y4_t o-> y4_t1']
   edges only with PYTHONHASHSEED=1: ['y0_t o-> y3_t', 'y2_t <-- y3_t', 'y3_t --> y3_t1', 'y3_t <-o y4_t', 'y4_t --> y4_t1']
   ```

   One adjacency appears (tremor–function in year $t$, a same-year one) and four edges change their marks. Over 16 synthetic fits (eight datasets, both configurations) run under six hash seeds, 1 graph changed and no lagged adjacency did; but nothing prevents a lagged adjacency from changing, which would change the support. Python draws a new hash seed for every process unless `PYTHONHASHSEED` is set, so two runs of the same script can differ, and the workers of `parallel.py` (spawned processes) each have their own seed. The claim "any number of workers gives identical results" (`parallel.py:6–7`; guide §0.6) therefore does not strictly hold for anything that runs Step 1: the support of the observed fits and of every bootstrap replicate, the separate graphs, C11. Setting `PYTHONHASHSEED` (for example to 0) in the environment before Python starts, which spawned workers then inherit, would restore it. The guide's own examples gave identical output under hash seeds 0–3.

3. **The exogenous rule differs from protocol §6.2.** The protocol says "nothing causes era, age or sex". The code ([`step1.py:150–154`](../../src/pdx/step1.py#L150), with `exempt=[era_col]` at [`step1.py:210`](../../src/pdx/step1.py#L210)) lets era cause age at onset and sex, and lets age at onset and sex cause each other; the module docstring says "nothing but era causes age at onset or sex". The protocol also says "age" where the code uses age at onset. One of the two texts should change.

**Documentation and protocol consistency**

4. **`depth` caps only the adjacency search.** The comment at [`step1.py:43`](../../src/pdx/step1.py#L43) ("separating-set cap") suggests a cap on every conditioning set, but Possible-D-SEP (`FCI.py:1000`) and R4 (`FCI.py:618`) test larger sets. Reproduction: [`ex15_depth_cap.py`](../guide_examples/s04/ex15_depth_cap.py) on a synthetic six-domain table:

   ```text
   depth=1: 1638 rows x 16 columns, 0.3 s; test calls by |S|: {0: 240, 1: 1250, 2: 51, 3: 65, 4: 40, 5: 21, 6: 7, 7: 1}
   depth=3: 1638 rows x 16 columns, 0.2 s; test calls by |S|: {0: 240, 1: 1249, 2: 250, 3: 76, 4: 34, 5: 10, 6: 1}
   ```

   (The counts can differ by a few calls between processes, for the reason in item 2.) The fit cost at ten domains therefore depends on more than `depth`; the engine gate's timing of Step 1 should use the real configuration.

5. **C5's conditioning-set cap.** `c5_tsldiffpc` defaults to `max_s=3` at every $d$ ([`run.py:63`](../../src/pdx/comparators/run.py#L63)). Protocol §8.4 and the tsLDiffPC/tsDCI note use every conditioning set at six domains and the cap only at ten domains; the note found that the cap changes power substantially (49% against 79% for a new link). The benchmark should pass `max_s=None` below ten domains, or the protocol should say otherwise.

6. **Comparator rows differ from Step 1's rows.** `_pairs` ([`run.py:19–30`](../../src/pdx/comparators/run.py#L19)) uses intervals of at most 1.5 years and complete visits; Step 1 uses grid years and test-wise deletion ([`data.py:117–121`](../../src/pdx/data.py#L117)). C11 mixes the two (its separate graphs go through `run_fci`, its interaction tests through `_pairs`). Worth one sentence in the benchmark's methods, or aligning both on grid years.

7. **`sortability` differs from Reisach's definition** ([`step1.py:259–282`](../../src/pdx/step1.py#L259)):
   - each ancestor pair counts once; Reisach et al. count a pair once per path length that connects it (their code multiplies the edge matrix by itself), so values on graphs with several paths are not comparable with published ones;
   - ties use exact equality, so after standardisation rounding noise decides. Reproduction: [`ex10_sortability.py`](../guide_examples/s04/ex10_sortability.py) gives var-sortability 0.000 on standardised data instead of 0.5. Reisach's code uses a relative tolerance of $10^{-9}$;
   - the protocol's "sortability baseline that ignores conditional independence" (§7.2), a learner that orders variables by variance or $R^2$, is not implemented.

8. **Not yet in code, though named in the protocol:** multiple imputation inside the tests (§7.2, Witte 2022); the cross-check learners S-FCI, tiered IOD and BOSS-FCI (§7.2); the speed-matched arm of C5 and C11 (§8.4); a harness that runs the comparators on the qualification datasets (only `scripts/smoke_test.py` calls them); wrappers for C7 and C10.

**Smaller points**

9. **"Cannot test" becomes "independent".** [`step1.py:81–82`](../../src/pdx/step1.py#L81) and [`:90–99`](../../src/pdx/step1.py#L90) return $p=1$ when $n\le k+3$, when a residual has zero variance, or when the sandwich variance is zero. FCI then removes the edge and records a separating set. Harmless at full size; possible in small subsamples or nearly constant columns. Reproduction: the last line of [`ex02_fisherz.py`](../guide_examples/s04/ex02_fisherz.py). Returning NaN and keeping the edge, or counting such tests, would be safer.

10. **The clustered test** ([`step1.py:93–99`](../../src/pdx/step1.py#L93)):
    - it always tests the slope of the higher-index column on the lower-index one, so it is not symmetric in $U$ and $V$ ([`ex03_cluster_test.py`](../guide_examples/s04/ex03_cluster_test.py): 0.395 against 0.369 when swapped). Natural for lagged pairs; arbitrary for same-year pairs;
    - the small-sample factor is $N_c/(N_c-1)$ only and the reference is the normal; statsmodels' CR1 also has $(n-1)/(n-p)$. Negligible at full size;
    - under strong shared heteroscedasticity the sandwich is still anti-conservative (9.8% at the 5% level with residual SDs $e^{0.8S}$, $n=300$; same example). WP3's calibration of the D9 candidates should include such a design.

11. **The interval rule is approximate with missing values** ([`step1.py:181–186`](../../src/pdx/step1.py#L181)). Each column is partialled on its own observed rows, not on each test's rows, and the Fisher-z statistic keeps $\sqrt{n-k-3}$ instead of $\sqrt{n-k-4}$. Exact on complete rows; no unit test covers `add_interval` (the code map lists none).

12. **`_orient_tiers` ignores required knowledge** ([`step1.py:119–128`](../../src/pdx/step1.py#L119)). Nothing adds required edges today; if anything does later, the replacement would drop them silently.

13. **tsLDiffPC's Meek rule 4 lacks a condition** ([`ts_ldiffpc.py:255–265`](../../src/pdx/comparators/ts_ldiffpc.py#L255)). Meek's R4 orients $y\to z$ only if $x$ and $z$ are non-adjacent; the code does not check it. Without the condition, the rule can orient an edge that other compatible graphs orient the other way. Found by reading, not reproduced; worth comparing with pyCIPHOD in the fidelity check (D23). Also, when two colliders disagree about one edge, the last one wins ([`ts_ldiffpc.py:214–228`](../../src/pdx/comparators/ts_ldiffpc.py#L214)).

14. **`c12_dynotears` ignores `converged`** ([`run.py:54`](../../src/pdx/comparators/run.py#L54)). A fit that stops with $h>10^{-8}$ is used silently; the benchmark could record the flag.

15. **Monte Carlo precision of the separate-graphs threshold** ([`separate.py:57`](../../src/pdx/separate.py#L57)). With 50 permutations the 95% quantile of the maxima is estimated from 50 values, and the observed $D$ shares one random stream with the permutations. Reporting the threshold's Monte Carlo error, or more permutations, would help; the toy run in §4.20 shows how rough a small reference is.

16. **What the tests do not cover.** `tests/test_step1.py` does not exercise `add_interval`, `include_dose`, the exogenous rule, `reported_map`, `sortability` or `separate.py`. `reported_map` and `separate_graph_reference` run only in `scripts/smoke_test.py`, without checks on their output; `sortability` is called nowhere. `test_no_edge_runs_backwards_in_time` cannot fail as long as tiers are used, because the tier arrowhead makes a year-$t+1$ → year-$t$ arrow impossible. `tests/test_comparators.py` checks only the shapes of the outputs; DYNOTEARS' recovery, the tsDCI and tsLDiffPC orientations and `gradient_check` are covered only by the toy validation scripts and this guide's examples.

17. **causal-learn behaviours to know** (version 0.1.4.8, pinned in `pyproject.toml` with a warning): the adjacency search stores the union of all separating sets found at the removal depth; R4 runs only in a pass where R1–R3 changed something; the selection-bias rules R5–R7 always run; `alpha` must be a Python `float` and `depth` a Python `int`; the function swap in `fci_graph` is process-wide (safe with the package's process-based parallelism, not with threads).

