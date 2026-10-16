# Air Quality Prediction from Environmental Data

This project estimates carbon monoxide concentration, `CO(GT)`, using air sensors, other pollutant measurements, weather, and the hour and day of the week. It compares Ridge, Lasso, and KNN on the UCI Air Quality dataset.

## Run it

Use Python 3.12 or later. The project was checked with Python 3.12.14. Download `AirQualityUCI.csv` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/360/air+quality) and put it in `data/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python air_quality_prediction.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For a shorter test run or a file stored elsewhere:

```bash
python air_quality_prediction.py --quick --jobs 2 --data /path/to/AirQualityUCI.csv
```

The script saves scores, test predictions, validation results, coefficients, plots, and run settings in `results/`. Use `--output` to choose another folder. The package versions in `requirements.txt` were used to check the project.

## How it works

Missing values are marked as `-200` in the data. The script removes rows with a missing target and leaves out `NMHC(GT)` because most of its values are missing. It fills missing inputs with training medians and scales the inputs inside each validation fold.

The split is 80% training and 20% test, with a fixed seed of 42. All three models use the same five validation folds and are tuned by RMSE. The model with the lowest validation RMSE is selected before the test scores are compared. The full search uses 100 Ridge penalties from 0.001 to 1,000, 100 Lasso penalties from 0.001 to 10, and 2 to 30 KNN neighbors with both weighting options and both distance measures. `--quick` reduces these grids.

This is a simple benchmark for estimating CO from measurements taken during the same hour. Some inputs come from reference analyzers, so the setup assumes those measurements are available. Nearby hours can fall on both sides of the random split. The scores therefore do not show how well a model would predict future pollution or work with low-cost sensors alone. That would need a different set of inputs and a separate test based on time.

The data comes from [Vito, S. (2008), Air Quality, UCI Machine Learning Repository](https://doi.org/10.24432/C59K5F). See [LICENSE.txt](LICENSE.txt) for the code license.
