# Air Quality Prediction from Environmental Data

This repository contains a statistical learning project on the UCI **Air Quality** dataset. The goal is to predict the reference carbon monoxide concentration, `CO(GT)`, from pollutant concentrations, low-cost sensor responses, and meteorological variables.

## Project overview

The analysis includes:

- data loading and cleaning;
- handling missing values encoded as `-200`;
- temporal interpolation of missing values;
- exploratory temporal analysis;
- correlation analysis and multicollinearity discussion;
- train-test split and cross-validation;
- Ridge regression, Lasso regression, and K-nearest neighbors;
- model comparison using RMSE, MAE, and R²;
- export of publication-ready figures.

## Dataset

Download the **Air Quality Data Set** from the UCI Machine Learning Repository and place the CSV file in:

```text
data/AirQualityUCI.csv
```

The notebook also supports a custom path through the environment variable `AIR_QUALITY_DATA`:

```bash
export AIR_QUALITY_DATA=/path/to/AirQualityUCI.csv
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Running the project

Run the notebook:

```bash
jupyter notebook air_quality_prediction.ipynb
```

Or run the script:

```bash
python air_quality_prediction.py
```

## Main results

The final models are evaluated on a held-out test set. In the obtained run, KNN with a small number of neighbors gives the best predictive performance, while Ridge and Lasso remain useful for interpretation because their standardized coefficients are easier to analyze. The KNN result should be interpreted carefully because very small values of `k` can be sensitive to noise and may require external validation.
