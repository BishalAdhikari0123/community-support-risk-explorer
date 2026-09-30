# Community Support Risk Explorer

A reproducible project for identifying where communities may face elevated cost-of-living pressure and where support teams could prioritise further investigation.

The project is designed to demonstrate the workflow expected of an entry-level data scientist:

- reproducible feature generation with a clear data dictionary
- leakage-aware train/test modelling
- probability calibration and threshold selection
- model interpretation through permutation importance and local explanations
- an interactive Streamlit dashboard for non-technical stakeholders
- tests for data quality and model behaviour
- an explicit ethics note covering proxy risk, uncertainty, and responsible use

> **Important:** the included dataset is a deterministic synthetic demo so the repository runs without credentials or a data licence. It is not a claim about any real local authority. The `data.py` module documents the replacement boundary for official ONS, DWP, HM Land Registry, and Department for Energy Security and Net Zero data.

## Project question

> Which local authorities should a council or charity prioritise for further investigation when planning cost-of-living support, and what factors are associated with the model's estimate?

This is a prioritisation aid, not an eligibility or individual-level decision system.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
streamlit run app.py
```

Run the tests:

```powershell
pytest
```

Rebuild the demo artefacts:

```powershell
python -m uk_risk.pipeline
```

## Repository map

```text
app.py                         Streamlit dashboard
src/uk_risk/data.py            synthetic demo data and schema checks
src/uk_risk/modeling.py        training, calibration, metrics, explanations
src/uk_risk/pipeline.py        reproducible command-line pipeline
tests/                         focused unit tests
reports/                       generated model card and metrics
```

## Method

The target is a simulated `high_pressure` flag at local-authority level. Features represent plausible aggregated indicators: median rent-to-income ratio, energy efficiency, unemployment, benefit reliance, overcrowding, and food insecurity. A logistic regression is used as the primary model because coefficients and calibrated probabilities are easier to communicate to policy stakeholders than a black-box score.

The evaluation uses a stratified holdout set and reports ROC-AUC, average precision, Brier score, and recall at a selected operating threshold. Since this is a prioritisation problem, average precision and recall are reported alongside ROC-AUC rather than relying on accuracy alone.

## Responsible use

- Do not infer hardship for an individual from this model.
- Area-level proxies can encode structural inequality and should not be used to deny services.
- Small-area estimates require confidence intervals and local validation before operational use.
- Any production version should use official, versioned sources, publish a data card, monitor drift, and involve affected communities.

## Extending it with official data

Replace `make_demo_dataset()` with an ingestion adapter that returns the same columns in `DATA_DICTIONARY`. For a UK deployment, suggested sources to investigate are the ONS annual population survey, ONS housing affordability statistics, DWP Stat-Xplore aggregates, the English Housing Survey, and DESNZ sub-national fuel poverty statistics. Check licences, geography revisions, suppression rules, and publication dates before joining them.
