<!-- markdownlint-disable MD033 MD041 -->

<table border="0">
  <tr>
    <td valign="bottom" border="0" width="75%">
      <h1>Time Series Toolkit</h1>
    </td>
    <td align="right" border="0">
      <img src="docs/megatron.png" alt="Megatron">
    </td>
  </tr>
</table>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/package-megatron-F28C28" alt="Package"></a>
  <a href="#test"><img src="https://img.shields.io/badge/coverage-88.86%25-2EA043" alt="Coverage"></a>
</p>

## About

<p align="justify">
<strong>Megatron</strong> is a Python package that provides a set of stepwise tools for processing, clustering and forecasting multiple time series. It also proposes a higher level orchestration modules to perform an end-to-end workflow.
</p>

## Project

### Purpose

<p align="justify">
It was originally intended to take part in <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting">Kaggle</a> retail time series forecasting problem, keep a simple notebook as a manual running solution and submit predictions. However diving deeper into this task the desire to build not only consistent but automated and scalable solution was appeared.
</p>

<p align="justify">
Proposed package components are almost designed to work with hierarchically structured (multiindex dataframe) time series e.g. <code>(store_id, product_id, date)</code> and aimed to:
</p>

- data cleaning, truncating, frequency recovering and missing values filling;
- series demand classification;
- demand-specific transformations;
- time series clustering by its shape and magnitude;
- forecasting strategies selection by its demand type and cluster size;
- strategy parameters tuning.

### Layout

```text
.
├── data/
│   ├── raw/                  # input CSV data
│   ├── processed/            # parquet data, was a raw before processing
│   ├── submissions/          # final forecast CSV output
│   └── test/                 # parquet synthetic and train sliced samples
├── docs/                     # graphic content
├── models/                   # persisted reusable objects
├── notebooks/                # disclose stepwise workflow validation files
├── scripts/
│   ├── process_raw_data.py   # raw to processed data running script
│   └── train_and_forecast.py # processed data to fitted models and forecast running script
├── src/megatron/
│   ├── clusterers/           # series clustering structures
│   ├── forecasters/          # forecasting strategy amd model instances
│   ├── pipelines/            # end-to-end orchestration modules
│   ├── transformers/         # preprocessing and feature extraction structures
│   ├── visualization/        # custom plotting modules
│   └── config.py             # global runtime configuration via constants
├── tests/
│   ├── unit_test.py
│   ├── integration_test.py
│   └── data_quality_test.py
├── Makefile
└── pyproject.toml   

```

## Architecture

The common pipeline follows two main stages:

1. preprocess raw retail datasets into aligned parquet artifacts;
2. train hierarchical forecasting pipelines on target variable, then export to a submission file.

![Pipeline diagram](docs/diagram.drawio.svg)

### `megatron.config`

<p align="justify">
Global configuration for the whole running session. Critical part of the correct pipeline setup &mdash; all required constants initialization (e.g. seasonal period, forecasting horizon or lag window limits) via <code>set_config(...)</code> before any of <code>megatron</code> modules import, otherwise the default values are set.
</p>

### `megatron.transformers`

<div align="justify">
  <p>Series transformation and feature engineering layer (most of them are multithreading, the other ones use vector calculations):</p>
  <ul>
    <li><code>InitialPreprocessing()</code>: removes useless series and trims leading or trailing zero-heavy segments;</li>
    <li><code>Mapper()</code>: maps data multiindex into a compact temporal <code>(index, date)</code> index and restores it after forecasting;</li>
    <li><code>DemandClassifier()</code>: splits all series into four demand classes — <code>smooth</code>, <code>erratic</code>, <code>intermittent</code>, and <code>lumpy</code>;</li>
    <li><code>PlateauDetector()</code>: finds long constant or missing consistent values and keeps only the data after them (works only for <code>smooth</code> and <code>erratic</code> demand classes respectively);</li>
    <li><code>ChangePointDetector()</code>: detects the most meaningful trend or variance break and keeps the latest history regime (works only for <code>smooth</code> and <code>erratic</code> demand classes respectively);</li>
    <li><code>OutlierDetector()</code>: for <code>smooth</code> and <code>erratic</code> demands <code>IsolationForest()</code> uses, otherwise &mdash; robust normalization with median and MAD applies (country holidays and promotion days are also taken into account);</li>
    <li><code>ExogenousDataTransformer()</code>: extracts calendar features such as holidays, weekends and wage days.</li>
  </ul>
</div>

### `megatron.clusterers`

<div align="justify">
  <p>Series clustering layer &mdash; groups series for further global forecasting (one model per series cluster), uses multithreading to define the most robust amount of clusters:</p>
  <ul>
    <li><code>SmoothErraticClusterer()</code>: for valid series with history length greater than required <code>KMeans</code> algorithms uses, otherwise &mdash; <code>WDTW</code> pairwise distance metric for invalid series applies in order to detect the closest valid one;</li>
    <li><code>IntermittentLumpyClusterer()</code>: standard <code>KMeans</code> algorithms uses.</li>
</div>

### `megatron.forecasters`

<div align="justify">
  <p>Forecasting execution layer:</p>
  <ul>
    <li><code>CommonForecaster()</code>: chooses a modeling strategy per cluster depending on its demand class and size &mdash; for big clusters the global (one per cluster) models uses, otherwise the local (one per series) model applies;</li>
    <li><code>se_complex_global</code>: global forecaster object with <code>recursive</code> training strategy, the core estimator is <code>LGBMRegressor()</code>;
    <li><code>se_simplex_global</code>: global forecaster object with <code>recursive</code> training strategy, the core estimator is <code>ElasticNet()</code>;
    <li><code>se_simplex_local</code>: local forecaster object, the core forecasters are <code>Prophet()</code> and <code>SARIMAX()</code>, the fallback one is <code>StatsForecastAutoTheta()</code>;
    <li><code>il_complex_global</code>: global forecaster object with <code>direct</code> training strategy, the core estimator is <code>LGBMRegressor()</code>;
    <li><code>il_simplex_global</code>: global forecaster object with <code>direct</code> training strategy, the core estimator is <code>HurdleModel()</code> (the product of <code>LogisticRegression()</code> and <code>ElasticNet()</code> estimators outputs);
    <li><code>il_simplex_local</code>: local forecaster object, the core forecasters are <code>Croston()</code>, <code>StatsForecastADIDA()</code> and <code>TSB()</code>;
    <li>Uses multithreading to optimize model parameters per its cluster independently;</li>
    <li>Persist all fitted models via <code>joblib</code> module into <code>models</code> directory to reuse them in subsequent runs.</li>
</div>

### `megatron.pipelines`

Orchestration layer:

- `CommonPipeline` runs transformation -> clustering -> forecasting for one demand class.
- `E2EForecaster` is the top-level entry point. It classifies demand first, then dispatches each subset to its matching `CommonPipeline`.

### `megatron.visualization`

Plotly-based helpers for:

- class-level series inspection;
- cluster visualization;
- anomaly/change-point inspection;
- train-vs-forecast comparison plots.

## Quick Start

### Installation

Recommended from project metadata:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,test]"
```

This installs:

- runtime dependencies from `pyproject.toml`;
- development tools such as `ruff`, `black`, `ipython`, and `ipykernel`;
- test tools such as `pytest` and `pytest-cov`.

### Data Setup

Place the expected raw CSV files from competition into `data/raw/` before start.

### Run

Using the Makefile:

```bash
make fit-all
```

Stepwise equivalent:

```bash
python scripts/process_raw_data.py   # same as make process-data
python scripts/train_and_forecast.py # same as make train-forecast
```

### Test

The test suite is split into three layers:

- `unit`: checks the individual components computational logic;
- `integration`: checks how package components work together;
- `data_quality`: validates processed dataset characteristics.

Run unit and integration tests with code coverage:

```bash
make test
```

Run data-quality checks without coverage:

```bash
make test-data
```

Run all:

```bash
make test-all
```

## Output Artifacts

After a successful run, the repository essentially complements with:

- processed parquet files in `data/processed/`;
- binary serialized clusterers and forecasting models in `models/`;
- final CSV submission file in `data/submissions/`.

## Suggested Documentation Additions

- a short section explaining how global vs local forecasters are selected;
- benchmark notes such as expected runtime on your machine;
