<!-- markdownlint-disable MD033 MD041 -->

<h1 align="center">Time Series Toolkit</h1>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/package-megatron-F28C28" alt="Package"></a>
  <a href="#test"><img src="https://img.shields.io/badge/coverage-88%25-2EA043" alt="Coverage"></a>
</p>

<p align="center">
  <img src="docs/megatron.png" alt="Megatron" width="150">
</p>

## About

<p align="justify">
<strong>Megatron</strong> is a Python package that provides a set of stepwise tools for processing, clustering and forecasting multiple time series. It also proposes several level orchestration modules to continuously perform an end-to-end workflow.
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

### Config

<p align="justify">
Global configuration for the whole running session. Critical part of the correct pipeline setup &mdash; all required constants initialization (e.g. seasonal period, forecasting horizon or lag window limits) via <code>set_config(...)</code> before any of <strong>megatron</strong> modules import, otherwise the default values are set.
</p>

### Pipelines

<div align="justify">
  <p>Orchestration and routing layers:</p>
  <ul>
    <li><code>E2EForecaster()</code>: highest &mdash; performs index mapping and classifies input processed data to a specific demand, finally delegates all further work to a next layer;</li>
    <li><code>CommonPipeline()</code>: middle one &mdash; consistently runs all demand-specific transformations, clustering and forecaster stages.</li>
</div>

### Transformers

<div align="justify">
  <p>Series transformation and feature engineering layer (most of them are multithreading, the other ones use vector calculations):</p>
  <ul>
    <li><code>InitialPreprocessing()</code>: removes useless series and trims leading or trailing zero-heavy segments;</li>
    <li><code>Mapper()</code>: maps data multiindex into a compact temporal <code>(index, date)</code> index and restores it after forecasting;</li>
    <li><code>DemandClassifier()</code>: splits all series into four demand classes — <i>smooth</i>, <i>erratic</i>, <i>intermittent</i>, and <i>lumpy</i>;</li>
    <li><code>PlateauDetector()</code>: finds long constant or missing consistent values and keeps only the data after them (works only for <i>smooth</i> and <i>erratic</i> demand classes respectively);</li>
    <li><code>ChangePointDetector()</code>: detects the most meaningful trend or variance break and keeps the latest history regime (works only for <i>smooth</i> and <i>erratic</i> demand classes respectively);</li>
    <li><code>OutlierDetector()</code>: for <i>smooth</i> and <i>erratic</i> demands <code>IsolationForest()</code> uses, otherwise &mdash; robust normalization with median and MAD applies (country holidays and promotion days are also taken into account);</li>
    <li><code>ExogenousDataTransformer()</code>: extracts calendar features such as holidays, weekends and wage days.</li>
  </ul>
</div>

### Clusterers

<div align="justify">
  <p>Series clustering layer &mdash; groups series for further global forecasting (one model per series cluster), uses multithreading to define the most robust amount of clusters:</p>
  <ul>
    <li><code>SmoothErraticClusterer()</code>: for valid series with history length greater than required <code>KMeans()</code> algorithms uses, otherwise &mdash; pairwise distance with <i>WDTW</i>-metric for invalid series applies in order to detect the closest valid one;</li>
    <li><code>IntermittentLumpyClusterer()</code>: standard <code>KMeans()</code> algorithms uses.</li>
</div>

### Forecasters

<div align="justify">
  <p>Forecasting execution layer:</p>
  <ul>
    <li><code>CommonForecaster()</code>: lowest orchestration layer &mdash; chooses a modeling strategy per cluster depending on its demand class and size, therefore for big clusters the global (one per cluster) models uses, otherwise the local (one per series) model applies. It also uses multithreading to tune models parameters simultaneously and persist all fitted models via <code>joblib</code> package into <code>models</code> directory to reuse them in subsequent runs;</li>
    <li><code>se_complex_global</code>: global forecaster with <i>recursive</i> strategy, the core estimator is <code>LGBMRegressor()</code>;
    <li><code>se_simplex_global</code>: global forecaster with <i>recursive</i> strategy, the core estimator is <code>ElasticNet()</code>;
    <li><code>se_simplex_local</code>: local forecaster, the core forecasters are <code>Prophet()</code> and <code>SARIMAX()</code>, the fallback one is <code>StatsForecastAutoTheta()</code>;
    <li><code>il_complex_global</code>: global forecaster with <i>direct</i> strategy, the core estimator is <code>LGBMRegressor()</code>;
    <li><code>il_simplex_global</code>: global forecaster with <i>direct</i> strategy, the core estimator is <code>HurdleModel()</code> (the product of <code>LogisticRegression()</code> and <code>ElasticNet()</code> estimators outputs);
    <li><code>il_simplex_local</code>: local forecaster, the core forecasters are <code>Croston()</code>, <code>StatsForecastADIDA()</code> and <code>TSB()</code>.
</div>

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
- development tools such as `ruff`, `black`, etc.;
- test tools &mdash; `pytest` and `pytest-cov`.

### Data Setup

Place the expected raw CSV files from <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting">Kaggle</a> into `data/raw` directory before start.

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

### Tests

The test suite is split into three layers:

- `unit`: checks the individual components computational logic;
- `integration`: checks how package components work together;
- `data_quality`: validates processed dataset characteristics.

Using the Makefile:

```bash
make test      # run unit and integration tests with code coverage
make test-data # run data-quality checks without coverage
make test-all
```

## Conclusion

### Tech notes

<div align="justify">
  <p>The e2e runtime workflow from raw data to final forecast was about <code>3h 56min</code> and performed on <code>Macbook Air (M2, 2022)</code> with <code>8GB RAM</code> and <code>8-core CPU</code>. After a successful run, the repository essentially complements with:</p>
    <ul>
      <li>processed parquet files in <code>data/processed</code>;
      <li>binary serialized clusterers and forecasters in <code>models</code>;
      <li>latest forecast CSV file in <code>data/submissions</code>.
</div>

<p align="justify">
Therefore instead of fitting a forecaster per series (<code>1638</code> time series in total) it was only necessary to successfully fit <code>4</code> clusterers and <code>54</code> forecasters.
</p>

### Kaggle achievements

As of latest submission the result was placed in leaderboard top `7%` :scream: :exploding_head: 
Thanks for **megatron** package :stuck_out_tongue_winking_eye:

## Future work

<div align="justify">
  <p>Here're several orchestration structure enhancements and computational components logic exploration to start with:</p>
    <ul>
      <li>a;
      <li>b;
      <li>c.
</div>