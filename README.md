# Predicting Victorian Road Crash Severity — A Multiclass Classification Study

A tidymodels-based analysis of the Victorian road crash dataset that predicts whether a crash results in a **Fatal**, **Serious**, or **Minor** injury outcome, using environmental, vehicle, and human factors. The final model is a **hierarchical XGBoost classifier** (Stage 1: Minor vs. Severe, Stage 2: Fatal vs. Serious among Severe crashes).

**[View the full report](https://dhruvlawaniya.github.io/victorian-crash-severity-classifier/)**

## Overview

- **Research question:** Which combinations of environmental (speed zone, lighting, weather, surface, geometry), vehicle (type, age, towing, occupancy), and human (age, sex, seatbelt/helmet use, ejection, licence origin) factors best predict Victorian road crash severity?
- **Data:** Victoria Road Crash Data, Department of Transport and Planning (2024), ~15,900 crashes/year, sourced from [discover.data.vic.gov.au](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data).
- **Target:** `SEVERITY` recoded into three classes — Fatal, Serious, Minor.
- **Stakeholders:** Ambulance Victoria (triage), VicRoads (infrastructure prioritisation), TAC (actuarial risk bands, education campaigns); findings are argued to transfer to NSW's Road Safety Action Plan.

## Repository contents

| File | Description |
|---|---|
| `eda_victoria_crash.Rmd` | R Markdown source: full EDA, preprocessing, modelling, and evaluation |
| `index.html` | Rendered/knitted HTML report (also served via GitHub Pages) |
| `README.md` | This file |

## Methodology

1. **EDA** — univariate/bivariate analysis vs. severity, class imbalance diagnosis (37:1), and a target-leakage check on `NO_PERSONS_KILLED`/`NO_PERSONS_INJ_*`.
2. **Preprocessing** — within-fold imputation, within-fold SMOTE (LR/KNN only), inverse-frequency case weighting for XGBoost, ordinal encoding for `SPEED_ZONE`, native factor handling for tree models.
3. **Models fitted** — penalised multinomial logistic regression, decision tree, random forest (`ranger`), weighted KNN (`kknn`), XGBoost, and a two-stage hierarchical XGBoost.
4. **Evaluation** — single held-out test-set pass, compared on macro-F1 and Fatal-class recall (not raw accuracy, given class imbalance).

## Key findings

- No single flat model dominates on both macro-F1 and Fatal recall — motivating the hierarchical XGBoost design.
- `SPEED_ZONE`, `ACCIDENT_TYPE`, `ROAD_GEOMETRY`, and `any_ejected` are the dominant predictors, consistent across Random Forest and XGBoost.
- Limitations: police-reported crashes only (biases Fatal proportion upward), some predictors unavailable at first dispatch, and no geographic stratification in cross-validation (candidate `LGA_NAME`-stratified CV for v2).

## Tech stack

R, tidymodels (`rsample`, `recipes`, `themis`, `yardstick`), `glmnet`, `rpart`, `ranger`, `xgboost`, `kknn`, `tidyverse`, `ggcorrplot`, `patchwork`, `vip`, knitr/R Markdown.

## Reproducing the report

```r
install.packages(c(
  "tidyverse", "scales", "lubridate", "patchwork", "knitr", "kableExtra",
  "ggcorrplot", "cowplot", "rsample", "recipes", "themis", "glmnet",
  "rpart", "ranger", "xgboost", "kknn", "yardstick", "vip"
))

rmarkdown::render("eda_victoria_crash.Rmd", output_file = "index.html")
```

The Victoria Road Crash dataset is not bundled in this repo (check licence/size before adding it) — download it from the [Victorian Government open data portal](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data) and update the data-loading chunk in `eda_victoria_crash.Rmd` with your local path.

