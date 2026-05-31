# ============================================================
# 16. K-nearest neighbors with grid search
# ============================================================

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
