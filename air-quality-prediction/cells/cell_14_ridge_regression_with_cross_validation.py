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
