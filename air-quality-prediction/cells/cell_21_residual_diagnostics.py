# ============================================================
# 21. Residual diagnostics
# ============================================================

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
