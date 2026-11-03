---
tags:
  - Training
  - EM
---

# Frequency-dependent training priors

`linker.training.estimate_parameters_using_pairwise_prior` is an optional,
experimental alternative to ordinary EM training. It estimates a match prior
that varies with the joint frequency of the values used to block the session.
This can supply information in sessions where few informative comparisons remain
after blocking. It does not guarantee identification or better linkage accuracy.

## Usage

After configuring a Linker and estimating U as usual:

```python
session = linker.training.estimate_parameters_using_pairwise_prior(
    ["first_name", "birthplace"],
    fix_u_probabilities=True,
    estimate_without_term_frequencies=False,
    max_pairs=10_000_000,
)

final = session.prior_history[-1]
print(session.converged)
print(final.coefficient, final.mean, final.minimum, final.maximum)
print(final.at_coefficient_bound)
```

Pass a nonempty list of plain input columns. The training rule matches exactly
on all of them. Precompute derived blocking keys as input columns; this API
does not accept arbitrary SQL blocking rules. Comparisons using the blocking
columns are excluded from training, as in ordinary EM. At least one comparison
must remain, and at least one of M or U must be free.

## Model

For each candidate pair, let `n_left` and `n_right` be the numbers of records
sharing its complete blocking tuple in its respective source datasets. Define:

```text
frequency_i = max(n_left_i, n_right_i)
z_i = -log(frequency_i) - center
logit(prior_i) = intercept + coefficient * z_i
```

Counts use the complete inputs, independently of candidate sampling. The center
is the mean negative log frequency weighted by candidate pairs used in training.
For deduplication within one source, both sides use that source's tuple count.

The coefficient starts at zero, giving a constant prior initially. A positive
coefficient assigns higher priors to rarer tuples; a negative coefficient reverses
the relationship. Every EM iteration fits the signed coefficient and intercept
using the current match responsibilities, alongside native comparison updates.
The intercept is calibrated so the pair-weighted mean prior equals the session's
estimated match proportion; it is not simply the logit of that mean.

The coefficient is unpenalised and bounded to `[-12, 12]` by default. Set
`prior_coefficient_bound` to change that absolute bound. The model does not impose
a one-to-one capacity constraint. In a complete cross-source tuple block, a
one-to-one assumption would imply an upper match fraction of `1/frequency`;
that motivates the frequency feature but is not enforced here.

U remains fixed by default. Setting `fix_u_probabilities=False` estimates session
U freely; it does not add U shrinkage. With
`estimate_without_term_frequencies=True`, exact counts are grouped by comparison
vector **and frequency** so the frequency information is retained. Otherwise,
the configured TF adjustments are applied during each EM iteration.

## Prediction and aggregation

The learned prior applies only to its training session. Prediction continues to
use the linker's global match prior, global comparison estimates, and configured
TF adjustments. Trained M/U estimates enter Splink's existing per-level median
aggregation. This estimator does not change that aggregation policy.

Pairwise-prior sessions are excluded when back-calculating the global prior from
ordinary scalar EM sessions. Their nonlinear context-specific prior does not
have the scalar interpretation required by that calculation.

## Diagnostics and limitations

`session.prior_history` returns a copy of the iteration history, starting with
iteration zero. Each immutable record contains:

| Attribute | Meaning |
| --- | --- |
| `iteration` | EM iteration, with zero for initial values |
| `mean` | Pair-weighted mean prior |
| `intercept`, `coefficient`, `center` | Parameters of the centered formula above |
| `minimum`, `maximum` | Prior range across observed training frequencies |
| `at_coefficient_bound` | Whether the coefficient equals its configured bound |

`session.converged` records whether the stopping criterion was met. The criterion
uses the maximum absolute change in comparison parameters and the prior
probabilities at observed frequencies. The inherited prior history chart shows
the session mean, rather than a separate curve for each frequency.

Warnings identify fewer than three distinct frequency contexts, a coefficient
at its bound, and an iteration limit reached without convergence. Constant
frequency cannot supply extra information; two frequency contexts may still
leave the mixture unidentified. Compare multiple initialisations and evaluate
linkage outcomes separately from convergence.

Intercept calibration retains relative accuracy for rare match or nonmatch
probabilities. An exact mean of zero or one raises `ValueError` rather than
silently replacing a collapsed mixture with a clipped probability.

The estimator retains the comparison model's assumptions. If M or U varies
with blocking context, the learned slope can partly absorb that mismatch.
Stability, plausible parameters, ranking quality, and probability calibration
are separate properties; none alone establishes accurate linkage.

See the [training API](../../api_docs/training.md) and
[session API](../../api_docs/em_training_session.md) for details.
