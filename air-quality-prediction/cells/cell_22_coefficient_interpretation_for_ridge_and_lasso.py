# ============================================================
# 22. Coefficient interpretation for Ridge and Lasso
# ============================================================

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
