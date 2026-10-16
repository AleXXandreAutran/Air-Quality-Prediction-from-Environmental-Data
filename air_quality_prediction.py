import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SEED = 42
TARGET = "CO(GT)"
FEATURES = [
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
    "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]
COLORS = ["#4477AA", "#228833", "#CC6677"]


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download AirQualityUCI.csv from UCI and place it in data/, "
            "or pass its path with --data."
        )
    data = pd.read_csv(path, sep=";", decimal=",").dropna(how="all")
    required = ["Date", "Time", TARGET, *FEATURES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    source_rows = len(data)
    data = data[required].copy()
    numeric = [TARGET, *FEATURES]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data[numeric] = data[numeric].replace([-200, np.inf, -np.inf], np.nan)
    missing_targets = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).copy()
    data["Datetime"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S", errors="coerce",
    )
    if data["Datetime"].isna().any():
        raise ValueError("Some rows with an observed target have an invalid date or time.")
    data = data.sort_values("Datetime").reset_index(drop=True)
    data["Hour"] = data["Datetime"].dt.hour
    data["DayOfWeek"] = data["Datetime"].dt.dayofweek
    if len(data) < 50:
        raise ValueError("At least 50 rows with an observed target are needed.")
    info = {"source_rows": source_rows, "missing_targets_removed": missing_targets}
    return data, info


def searches(quick, jobs):
    alphas_ridge = np.logspace(-3, 3, 5 if quick else 100)
    alphas_lasso = np.logspace(-3, 1, 5 if quick else 100)
    models = {
        "Ridge": (Ridge(), {"model__alpha": alphas_ridge}),
        "Lasso": (Lasso(max_iter=50000), {"model__alpha": alphas_lasso}),
        "KNN": (
            KNeighborsRegressor(n_jobs=1),
            {
                "model__n_neighbors": [2, 5, 10] if quick else list(range(2, 31)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    result = {}
    for name, (model, grid) in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        result[name] = GridSearchCV(
            pipeline, grid, cv=cv, scoring="neg_root_mean_squared_error",
            n_jobs=jobs, error_score="raise",
        )
    return result


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(x_train, y_train, metrics, predictions, fitted, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    names = list(fitted)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    labels = {"test_rmse": "Test RMSE (mg/m³)", "test_mae": "Test MAE (mg/m³)", "test_r2": "Test R²"}
    by_name = metrics.set_index("model").loc[names]
    for ax, (metric, label) in zip(axes, labels.items()):
        values = by_name[metric].to_numpy()
        bars = ax.bar(names, values, color=COLORS)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylabel(label)
        ax.margins(y=0.15)
    save_figure(fig, figures / "model_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    low = float(predictions[["observed", *names]].min().min())
    high = float(predictions[["observed", *names]].max().max())
    for ax, name, color in zip(axes, names, COLORS):
        ax.scatter(predictions["observed"], predictions[name], s=10, alpha=0.35, color=color)
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set(title=name, xlabel="Observed CO (mg/m³)")
    axes[0].set_ylabel("Estimated CO (mg/m³)")
    save_figure(fig, figures / "predicted_vs_observed.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, name, color in zip(axes, names, COLORS):
        ax.scatter(predictions[name], predictions["observed"] - predictions[name],
                   s=10, alpha=0.35, color=color)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(title=name, xlabel="Estimated CO (mg/m³)")
    axes[0].set_ylabel("Observed − estimated (mg/m³)")
    save_figure(fig, figures / "residuals.png")

    corr = x_train.assign(**{TARGET: y_train}).corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title("Correlations in the training data")
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    save_figure(fig, figures / "training_correlations.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cv_results = pd.DataFrame(fitted["KNN"].cv_results_)
    for (weights, p), group in cv_results.groupby(["param_model__weights", "param_model__p"]):
        group = group.sort_values("param_model__n_neighbors")
        distance = "Manhattan" if p == 1 else "Euclidean"
        ax.plot(group["param_model__n_neighbors"].astype(int), -group["mean_test_score"],
                marker="o", markersize=3, label=f"{weights}, {distance}")
    ax.set(xlabel="Number of neighbors", ylabel="Validation RMSE (mg/m³)",
           title="KNN validation results")
    ax.legend()
    save_figure(fig, figures / "knn_validation.png")

    coefficients = pd.DataFrame({
        name: fitted[name].best_estimator_.named_steps["model"].coef_
        for name in ["Ridge", "Lasso"]
    }, index=x_train.columns)
    coefficients.index.name = "feature"
    coefficients.to_csv(output / "coefficients.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    order = coefficients["Ridge"].abs().sort_values().index
    for ax, name, color in zip(axes, ["Ridge", "Lasso"], COLORS):
        ax.barh(order, coefficients.loc[order, name], color=color)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title=name, xlabel="Coefficient after scaling")
    save_figure(fig, figures / "coefficients.png")


def run(args):
    data_path = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    data, info = load_data(data_path)
    x = data[[*FEATURES, "Hour", "DayOfWeek"]]
    y = data[TARGET]
    train_index, test_index = train_test_split(
        data.index, test_size=0.2, random_state=SEED,
    )
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    output.mkdir(parents=True, exist_ok=True)
    missing = pd.DataFrame({"training": x_train.isna().sum(), "test": x_test.isna().sum()})
    missing.index.name = "feature"
    missing.to_csv(output / "missing_values.csv")
    fitted = searches(args.quick, args.jobs)
    for name, search in fitted.items():
        print(f"Training {name}...", flush=True)
        search.fit(x_train, y_train)
        pd.DataFrame(search.cv_results_).to_csv(output / f"cv_results_{name.lower()}.csv", index=False)
    selected = min(fitted, key=lambda name: -fitted[name].best_score_)
    rows = []
    predictions = data.loc[test_index, ["Datetime", TARGET]].rename(columns={TARGET: "observed"})
    for name, search in fitted.items():
        predicted = search.predict(x_test)
        predictions[name] = predicted
        params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
        rows.append({
            "model": name,
            "cv_rmse": -search.best_score_,
            "test_rmse": root_mean_squared_error(y_test, predicted),
            "test_mae": mean_absolute_error(y_test, predicted),
            "test_r2": r2_score(y_test, predicted),
            "parameters": json.dumps(params, sort_keys=True),
        })
    metrics = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.sort_values("Datetime").to_csv(output / "predictions.csv", index=False)
    info.update({
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows_used": len(data), "training_rows": len(train_index), "test_rows": len(test_index),
        "first_measurement": data["Datetime"].min().isoformat(),
        "last_measurement": data["Datetime"].max().isoformat(),
        "features": x.columns.tolist(), "target": TARGET, "seed": SEED,
        "split": "Random 80/20 split; 5 shuffled folds on training data only",
        "imputation": "Median fitted within each training fold; missing targets are removed",
        "selection_metric": "Lowest mean cross-validated RMSE",
        "selected_model": selected, "quick": args.quick,
        "task": "Estimate CO from measurements from the same hour",
        "limitation": "Nearby hours can appear in training and test data. This does not measure future forecasting performance.",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
            "scikit-learn": sklearn.__version__,
        },
    })
    (output / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    make_plots(x_train, y_train, metrics, predictions, fitted, output)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by validation RMSE: {selected}")
    print(f"Saved results to {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare three models for same-hour CO estimation.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "AirQualityUCI.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel grid-search workers; -1 uses all cores.")
    parser.add_argument("--quick", action="store_true", help="Use smaller grids for a short test run.")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")
    try:
        run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
