# 2. Load the dataset

from pathlib import Path
import os
import pandas as pd

# GitHub-friendly configuration:
# - Put AirQualityUCI.csv in the ./data folder, or
# - Set the AIR_QUALITY_DATA environment variable to the CSV path.
DATA_DIR = Path("data")
CSV_FILE = "AirQualityUCI.csv"

DATA_PATH = Path(os.environ.get("AIR_QUALITY_DATA", DATA_DIR / CSV_FILE))

# If the default file is not found, try to load the first CSV file in ./data.
if not DATA_PATH.exists():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if csv_files:
        DATA_PATH = csv_files[0]
    else:
        raise FileNotFoundError(
            f"Dataset not found at:
{DATA_PATH}

"
            "Please download AirQualityUCI.csv from the UCI Machine Learning Repository "
            "and place it in the ./data folder, or set AIR_QUALITY_DATA to the full CSV path."
        )

print(f"Loading dataset from: {DATA_PATH}")
raw_df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
raw_df = raw_df.dropna(axis=1, how="all")  # Remove empty trailing columns

print(f"Initial shape: {raw_df.shape}")
display(raw_df.head())
