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
