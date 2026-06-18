# 18. Summary table

results = pd.DataFrame([
    {"Model": "Ridge", **ridge_metrics, "Best hyperparameters": f"alpha={ridge_alpha:.4f}"},
    {"Model": "Lasso", **lasso_metrics, "Best hyperparameters": f"alpha={lasso_alpha:.4f}"},
    {"Model": "KNN", **knn_metrics, "Best hyperparameters": str(knn_search.best_params_)},
])

results = results.sort_values("RMSE").reset_index(drop=True)

display(results.round({"RMSE": 4, "MAE": 4, "R2": 4}))
