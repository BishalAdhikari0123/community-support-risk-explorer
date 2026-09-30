# Model card: Community Support Risk Explorer

## Intended use

Prioritise local authority areas for further investigation by councils, charities, or researchers. The output is an area-level screening signal.

## Out of scope

Individual eligibility, enforcement, fraud detection, service denial, or claims about any real authority in this repository.

## Data

The checked-in workflow generates 180 deterministic synthetic records using plausible ranges. No real person or local authority is represented. A production implementation needs official sources, source dates, geography keys, missingness treatment, and a reproducible join log.

## Model

A class-balanced logistic regression with standardised numerical features. The holdout set is stratified and fixed with seed 42. Metrics include ROC-AUC, average precision, Brier score, and recall at a 0.50 threshold.

## Risks and mitigations

- **Proxy discrimination:** housing and employment variables can reflect structural disadvantage. Review outputs with affected communities and never automate access decisions.
- **Ecological fallacy:** area averages do not describe every resident. Keep the unit of analysis visible in the UI.
- **Drift:** costs, labour markets, and policy change. Refit and monitor performance by nation and geography.
- **Uncertainty:** display probabilities as estimates and retain human review; production use should add confidence intervals and sensitivity analysis.
