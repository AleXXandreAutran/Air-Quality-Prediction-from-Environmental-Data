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
