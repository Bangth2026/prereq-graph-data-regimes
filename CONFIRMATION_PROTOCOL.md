# Confirmation protocol (fixed before any confirmation data are obtained)

Purpose: confirm or refute, on data that no analysis in this package has touched, the post hoc findings of the
manuscript. Eligible data: a later export of the Junyi Academy platform, or another platform's log with
prerequisite metadata. Neither is in hand at the time of writing.

## Fixed settings

- Cases: first attempt at a target never attempted before, at least one candidate prerequisite, learner with
  at least 5 attempts before the window. Positional split 60/20/20 per learner.
- Protocols: snapshot and fully sequential (Table 2 of the manuscript).
- Estimators for unattempted prerequisites: ZERO, POP, BKT prior, cold-start logistic KT, DKT, each fitted on
  training segments only with the settings of `bkt_prior.py`, `benchmark.py` and `kt_dkt.py`.
- Uses: pooled Eq. (1), ESD (Eq. 2–3).
- Learner models: B1 and B1RK (B1 + recency + DKT target prediction).
- Graph variants: B5 (graded), B9 (unweighted candidate list).
- Model: L2 logistic regression (C = 1, lbfgs, max_iter = 500) on standardised features, fitted on validation
  cases, scored once on held-out cases. Metric: held-out ΔAUC with a 300-replicate learner bootstrap (models
  fixed) and a DeLong test; failures flagged per 1,000 cases at a 20 % alert budget.

## Hypotheses and decision rules

H1. Under the snapshot protocol, the zero convention leaves at least 60 % of (case, prerequisite) pairs
    unobserved and supplies at least 90 % of the pooled Eq. (1) score mass. Confirmed if both hold.
H2. ESD with ZERO exceeds pooled Eq. (1) with ZERO on B5 over its own baseline. Confirmed if the
    ESD increment over B1 + counts is above the pooled increment over B1 and the ESD interval lies above 0.
H3. DKT is the most accurate estimator for unattempted prerequisites. Confirmed if its Brier score is the
    lowest of the five on later-attempted pairs.
H4. Over B1RK, ESD-DKT on B5 adds less than 0.005 AUC under the fully sequential protocol. Confirmed if the
    upper end of the interval is below 0.005.
H5. Graded weights do not beat the unweighted list. An advantage of grading is claimed only if the B5 − B9
    interval lies entirely above 0 under ESD with DKT over B1RK in the fully sequential protocol.

## Data-regime study (added in package version 2.1.0)

These hypotheses bind future data only. They were written after the data-regime results reported in the manuscript
were known, so those results are not a test of them.

Settings: as above, plus the data-regime procedure of `data_regime.sh` and `gpe2.py`: training learners subsampled at
0.5, 1, 2, 3.2, 5, 10, 25 and 100% (three samples each); DKT retrained to early stopping (patience 2) on the sampled
learners; difficulty, frequency, POP and LKT refitted on them; validation learners subsampled in the same proportion
(every layer fitted on the sampled learners only); held-out window unchanged. Increment = ESD-DKT on B5 over B1RK.
Pooled 90% interval = 5th–95th percentile of the seed-by-learner-bootstrap mixture.

H6. The increment decreases with the number of training learners. Confirmed if the slope of the weighted
    least-squares fit of the increment on log10(training learners) has a 95% interval entirely below 0.
H7. At the smallest size the increment exceeds the topic- and observation-matched random-relation control.
    Confirmed if the real-minus-random difference at 0.5% (or at the smallest size with at least 500 learners)
    has a pooled 95% interval above 0.
H8. The increment is equivalent to zero within ±0.005 (TOST, pooled 90% interval inside ±0.005) at every size
    of at least 10,000 training learners, and exceeds +0.005 (pooled 90% interval above +0.005) at no size of
    at least 10,000 training learners.
H9. GPE does not improve held-out AUC over Platt recalibration of its base estimator by more than 0.005 for
    any base. Confirmed if every GPE minus Platt AUC difference is below 0.005.

No further specification search on the present held-out window is planned.
