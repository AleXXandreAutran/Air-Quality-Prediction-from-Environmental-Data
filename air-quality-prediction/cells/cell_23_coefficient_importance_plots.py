# ============================================================
# 23. Coefficient importance plots
# ============================================================

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
