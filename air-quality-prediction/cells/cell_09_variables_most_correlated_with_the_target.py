# ============================================================
# 9. Variables most correlated with the target
# ============================================================

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
