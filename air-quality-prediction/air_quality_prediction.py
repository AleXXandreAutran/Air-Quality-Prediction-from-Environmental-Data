# 1. Libraries and global configuration

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Plotting style
sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette="viridis",
    font_scale=1.05
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def save_figure(name: str) -> None:
    """Save the current Matplotlib figure as both PDF and PNG."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")

# 2. Load the dataset

from pathlib import Path
import os
import pandas as pd

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


# 3. Variable dictionary and basic information

variable_descriptions = {
    "Date": "Measurement date",
    "Time": "Measurement time",
    "CO(GT)": "Reference carbon monoxide concentration",
    "PT08.S1(CO)": "Sensor 1 response, mainly sensitive to CO",
    "NMHC(GT)": "Non-methane hydrocarbons",
    "C6H6(GT)": "Reference benzene concentration",
    "PT08.S2(NMHC)": "Sensor 2 response, mainly sensitive to NMHC",
    "NOx(GT)": "Reference nitrogen oxides concentration",
    "PT08.S3(NOx)": "Sensor 3 response, mainly sensitive to NOx",
    "NO2(GT)": "Reference nitrogen dioxide concentration",
    "PT08.S4(NO2)": "Sensor 4 response, mainly sensitive to NO2",
    "PT08.S5(O3)": "Sensor 5 response, mainly sensitive to ozone",
    "T": "Temperature",
    "RH": "Relative humidity",
    "AH": "Absolute humidity",
}

summary_info = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": [raw_df[col].dtype for col in raw_df.columns],
    "unique_values": [raw_df[col].nunique(dropna=True) for col in raw_df.columns],
    "description": [variable_descriptions.get(col, "No description available") for col in raw_df.columns],
})

display(summary_info)


# 4. Cleaning missing values and creating a datetime variable

df = raw_df.copy()

# Replace the dataset-specific missing value code by NaN
numeric_candidates = [col for col in df.columns if col not in ["Date", "Time"]]
df[numeric_candidates] = df[numeric_candidates].replace(-200, np.nan)

# Combine Date and Time into a single datetime column
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

df = df.sort_values("Datetime").reset_index(drop=True)

missing_before = df.isna().sum().sort_values(ascending=False)
print("Missing values before imputation:")
display(missing_before[missing_before > 0].to_frame("missing_count"))

# Remove NMHC(GT), which is mostly missing
df = df.drop(columns=["NMHC(GT)"], errors="ignore")

# Target variable
TARGET = "CO(GT)"

# Remove rows without target values
df = df.dropna(subset=[TARGET]).copy()

# Interpolate remaining numeric missing values in chronological order
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

missing_after = df.isna().sum().sort_values(ascending=False)

print(f"Shape after cleaning: {df.shape}")
print("\nRemaining missing values:")
display(missing_after[missing_after > 0].to_frame("missing_count"))

display(df.head())


# 5. Create useful time variables for exploratory analysis

df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek  # Monday = 0, Sunday = 6
df["DayName"] = df["Datetime"].dt.day_name()

ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["DayName"] = pd.Categorical(df["DayName"], categories=ordered_days, ordered=True)

display(df[numeric_cols].describe().T.round(3))


# 6. Daily cycles: hourly boxplots

variables_to_plot = [
    "CO(GT)", "PT08.S1(CO)", "C6H6(GT)",
    "NOx(GT)", "NO2(GT)", "T", "RH", "AH"
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
axes = axes.ravel()

for ax, var in zip(axes, variables_to_plot):
    sns.boxplot(
        data=df,
        x="Hour",
        y=var,
        ax=ax,
        color=sns.color_palette("viridis", 8)[3],
        fliersize=0.6,
        linewidth=0.8
    )
    ax.set_title(var)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00", "06", "12", "18", "23"])

fig.suptitle("Hourly distribution of pollutants and meteorological variables", y=1.03, fontsize=16, fontweight="bold")
save_figure("hourly_cycles_boxplots")
plt.show()


# 7. Weekly average profile

weekly_means = (
    df.groupby("DayName", observed=True)[variables_to_plot]
    .mean()
    .reset_index()
)

weekly_long = weekly_means.melt(
    id_vars="DayName",
    var_name="Variable",
    value_name="Mean value"
)

g = sns.FacetGrid(
    weekly_long,
    col="Variable",
    col_wrap=4,
    sharey=False,
    height=3.0,
    aspect=1.25
)
g.map_dataframe(
    sns.lineplot,
    x="DayName",
    y="Mean value",
    marker="o",
    linewidth=2
)
g.set_titles("{col_name}")
g.set_axis_labels("", "Mean value")

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

g.fig.suptitle("Average weekly profile", y=1.04, fontsize=16, fontweight="bold")
g.fig.tight_layout()
g.fig.savefig(FIG_DIR / "weekly_profile.pdf", bbox_inches="tight")
g.fig.savefig(FIG_DIR / "weekly_profile.png", bbox_inches="tight")
plt.show()


# 8. Correlation analysis

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(13, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson correlation"}
)
plt.title("Correlation matrix after cleaning")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
save_figure("correlation_matrix")
plt.show()


# 9. Variables most correlated with the target

target_corr = (
    corr_matrix[TARGET]
    .drop(labels=[TARGET], errors="ignore")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

display(target_corr.to_frame("Correlation with CO(GT)").round(3))

plt.figure(figsize=(9, 6))
sns.barplot(
    x=target_corr.abs().head(10).values,
    y=target_corr.abs().head(10).index,
    palette="viridis"
)
plt.xlabel("Absolute Pearson correlation")
plt.ylabel("")
plt.title("Top variables associated with CO(GT)")
save_figure("top_target_correlations")
plt.show()

# 10. Scatterplots of the most correlated variables

top_features = target_corr.head(5).index.tolist()

fig, axes = plt.subplots(1, len(top_features), figsize=(4.2 * len(top_features), 3.8), sharey=True)

for ax, feature in zip(axes, top_features):
    sns.scatterplot(
        data=df.sample(min(len(df), 3000), random_state=RANDOM_STATE),
        x=feature,
        y=TARGET,
        ax=ax,
        alpha=0.35,
        s=18,
        edgecolor=None
    )
    sns.regplot(
        data=df.sample(min(len(df), 3000), random_state=RANDOM_STATE),
        x=feature,
        y=TARGET,
        ax=ax,
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.6}
    )
    ax.set_title(feature)
    ax.set_xlabel(feature)

axes[0].set_ylabel(TARGET)
fig.suptitle("Relationship between CO(GT) and the most correlated predictors", y=1.04, fontsize=15, fontweight="bold")
save_figure("scatter_top_correlations")
plt.show()


# 11. Simple outlier screening using the IQR rule

outlier_summary = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary.append({
        "variable": col,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "potential_outliers": count,
        "percentage": 100 * count / len(df)
    })

outlier_summary = pd.DataFrame(outlier_summary).sort_values("potential_outliers", ascending=False)
display(outlier_summary.round(3))

strong_pairs = []
cols = corr_matrix.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        value = corr_matrix.iloc[i, j]
        if abs(value) > 0.90:
            strong_pairs.append({
                "Variable 1": cols[i],
                "Variable 2": cols[j],
                "Correlation": value
            })

strong_pairs = pd.DataFrame(strong_pairs).sort_values("Correlation", key=lambda s: s.abs(), ascending=False)
display(strong_pairs.round(3))


# 12. Feature matrix and target vector

features_to_drop = [TARGET, "Date", "Time", "Datetime", "DayName"]
X = df.drop(columns=features_to_drop, errors="ignore")
y = df[TARGET]

# Keep only numerical features for the models
X = X.select_dtypes(include=[np.number]).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape:     {X_test.shape}")

display(X_train.head())


# 13. Helper functions for model evaluation

def regression_metrics(y_true, y_pred) -> dict:
    """Compute standard regression metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def print_metrics(model_name: str, metrics: dict) -> None:
    """Pretty-print model metrics."""
    print(f"\n{model_name}")
    print("-" * len(model_name))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# 14. Ridge regression with cross-validation

ridge_alphas = np.logspace(-3, 3, 100)

ridge_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=ridge_alphas, cv=cv))
])

ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

ridge_metrics = regression_metrics(y_test, y_pred_ridge)
ridge_alpha = ridge_model.named_steps["model"].alpha_

print(f"Best Ridge alpha: {ridge_alpha:.4f}")
print_metrics("Ridge regression", ridge_metrics)


# 15. Lasso regression with cross-validation

lasso_alphas = np.logspace(-3, 1, 100)

lasso_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LassoCV(
        alphas=lasso_alphas,
        cv=cv,
        max_iter=50000,
        random_state=RANDOM_STATE
    ))
])

lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

lasso_metrics = regression_metrics(y_test, y_pred_lasso)
lasso_alpha = lasso_model.named_steps["model"].alpha_

lasso_coefs = pd.Series(
    lasso_model.named_steps["model"].coef_,
    index=X_train.columns
)

selected_lasso_features = (
    lasso_coefs[lasso_coefs != 0]
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

print(f"Best Lasso alpha: {lasso_alpha:.4f}")
print_metrics("Lasso regression", lasso_metrics)

print("\nNumber of selected variables:", selected_lasso_features.shape[0])
display(selected_lasso_features.to_frame("Lasso coefficient").round(4))


# 16. K-nearest neighbors with grid search

knn_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor())
])

knn_grid = {
    "model__n_neighbors": list(range(2, 31)),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  # Manhattan and Euclidean distances
}

knn_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    return_train_score=True
)

knn_search.fit(X_train, y_train)

knn_model = knn_search.best_estimator_
y_pred_knn = knn_model.predict(X_test)

knn_metrics = regression_metrics(y_test, y_pred_knn)

print("Best KNN parameters:")
print(knn_search.best_params_)
print_metrics("KNN regression", knn_metrics)


# 17. Influence of k on KNN performance

knn_cv_results = pd.DataFrame(knn_search.cv_results_)
knn_cv_results["RMSE_CV"] = -knn_cv_results["mean_test_score"]

# Select Euclidean + uniform results for a clean k-curve
k_curve = knn_cv_results[
    (knn_cv_results["param_model__weights"] == "uniform") &
    (knn_cv_results["param_model__p"] == 2)
].copy()

k_curve["k"] = k_curve["param_model__n_neighbors"].astype(int)

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=k_curve.sort_values("k"),
    x="k",
    y="RMSE_CV",
    marker="o",
    linewidth=2.2
)
plt.axvline(
    x=knn_search.best_params_["model__n_neighbors"],
    linestyle="--",
    linewidth=1.5,
    color="black",
    label=f"Best k = {knn_search.best_params_['model__n_neighbors']}"
)
plt.xlabel("Number of neighbors k")
plt.ylabel("Cross-validated RMSE")
plt.title("Influence of k on KNN performance")
plt.legend()
save_figure("knn_k_selection")
plt.show()


# 18. Summary table

results = pd.DataFrame([
    {"Model": "Ridge", **ridge_metrics, "Best hyperparameters": f"alpha={ridge_alpha:.4f}"},
    {"Model": "Lasso", **lasso_metrics, "Best hyperparameters": f"alpha={lasso_alpha:.4f}"},
    {"Model": "KNN", **knn_metrics, "Best hyperparameters": str(knn_search.best_params_)},
])

results = results.sort_values("RMSE").reset_index(drop=True)

display(results.round({"RMSE": 4, "MAE": 4, "R2": 4}))


# 19. Visual comparison of model performance

metrics_long = results.melt(
    id_vars="Model",
    value_vars=["RMSE", "MAE", "R2"],
    var_name="Metric",
    value_name="Value"
)

g = sns.catplot(
    data=metrics_long,
    x="Model",
    y="Value",
    col="Metric",
    kind="bar",
    sharey=False,
    height=4,
    aspect=1.05,
    palette="viridis"
)

g.set_axis_labels("", "Value")
g.set_titles("{col_name}")

for ax in g.axes.flat:
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    # Add value labels above bars.
    # This version is compatible with older matplotlib versions
    # where ax.bar_label() does not exist.
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()

            if pd.notna(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

g.fig.suptitle(
    "Comparison of model performance on the test set",
    y=1.08,
    fontsize=16,
    fontweight="bold"
)

g.fig.tight_layout()
g.fig.savefig(FIG_DIR / "model_performance_comparison.pdf", bbox_inches="tight")
g.fig.savefig(FIG_DIR / "model_performance_comparison.png", bbox_inches="tight")
plt.show()


# 20. Predicted versus true values

prediction_df = pd.DataFrame({
    "True value": y_test,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "KNN": y_pred_knn
})

prediction_long = prediction_df.melt(
    id_vars="True value",
    var_name="Model",
    value_name="Predicted value"
)

g = sns.FacetGrid(
    prediction_long,
    col="Model",
    height=4,
    aspect=1,
    sharex=True,
    sharey=True
)

g.map_dataframe(
    sns.scatterplot,
    x="True value",
    y="Predicted value",
    alpha=0.35,
    s=18,
    edgecolor=None
)

# Add the ideal diagonal to each panel
min_val = min(prediction_long["True value"].min(), prediction_long["Predicted value"].min())
max_val = max(prediction_long["True value"].max(), prediction_long["Predicted value"].max())

for ax in g.axes.flat:
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.4)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)

g.set_axis_labels("True CO(GT)", "Predicted CO(GT)")
g.set_titles("{col_name}")
g.fig.suptitle("Predicted versus true CO(GT)", y=1.08, fontsize=16, fontweight="bold")
g.fig.tight_layout()
g.fig.savefig(FIG_DIR / "predictions_vs_true.pdf", bbox_inches="tight")
g.fig.savefig(FIG_DIR / "predictions_vs_true.png", bbox_inches="tight")
plt.show()


# 21. Residual diagnostics

residuals_df = pd.DataFrame({
    "Ridge": y_test - y_pred_ridge,
    "Lasso": y_test - y_pred_lasso,
    "KNN": y_test - y_pred_knn
}).melt(var_name="Model", value_name="Residual")

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=residuals_df,
    x="Model",
    y="Residual",
    palette="viridis",
    width=0.55,
    showfliers=False
)
sns.stripplot(
    data=residuals_df.sample(min(len(residuals_df), 2500), random_state=RANDOM_STATE),
    x="Model",
    y="Residual",
    color="black",
    alpha=0.12,
    size=2
)
plt.axhline(0, linestyle="--", linewidth=1.3, color="black")
plt.title("Residual distribution by model")
plt.xlabel("")
plt.ylabel("Residual = true value - predicted value")
save_figure("residual_distribution")
plt.show()


# 22. Coefficient interpretation for Ridge and Lasso

ridge_coefs = pd.Series(
    ridge_model.named_steps["model"].coef_,
    index=X_train.columns
).sort_values(key=lambda s: s.abs(), ascending=False)

lasso_coefs_sorted = lasso_coefs.sort_values(key=lambda s: s.abs(), ascending=False)

coef_table = pd.DataFrame({
    "Ridge coefficient": ridge_coefs,
    "Lasso coefficient": lasso_coefs_sorted
}).fillna(0)

display(coef_table.round(4))


# 23. Coefficient importance plots

top_n = 12

ridge_top = ridge_coefs.head(top_n).sort_values()
lasso_top = lasso_coefs_sorted.head(top_n).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(
    x=ridge_top.values,
    y=ridge_top.index,
    ax=axes[0],
    palette="viridis"
)
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Top Ridge coefficients")
axes[0].set_xlabel("Coefficient value")
axes[0].set_ylabel("")

sns.barplot(
    x=lasso_top.values,
    y=lasso_top.index,
    ax=axes[1],
    palette="viridis"
)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Top Lasso coefficients")
axes[1].set_xlabel("Coefficient value")
axes[1].set_ylabel("")

fig.suptitle("Linear model coefficients after standardization", y=1.03, fontsize=16, fontweight="bold")
save_figure("linear_model_coefficients")
plt.show()


# 24. Final 

best_model_name = results.iloc[0]["Model"]

print(f"Best predictive model according to RMSE: {best_model_name}")
print("\nInterpretation:")
print(
    "- KNN is recommended for pure prediction because it achieves the lowest error.\n"
    "- Ridge is recommended for interpretation because it handles multicollinearity "
    "while keeping readable coefficients.\n"
    "- Lasso does not perform strong variable selection here because the optimal alpha is very small."
)

