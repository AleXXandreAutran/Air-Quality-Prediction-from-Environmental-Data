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
