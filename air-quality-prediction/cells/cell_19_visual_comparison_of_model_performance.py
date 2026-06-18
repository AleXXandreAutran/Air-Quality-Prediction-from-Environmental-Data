# 19. Visual comparison of model performance

metrics_long = results.melt(
    id_vars="Model",
    value_vars=["RMSE", "MAE", "R2"],
    var_name="Metric",
    value_name="Value"
)

g = sns.catplot(
    data=metrics_long,
    x="Model",
    y="Value",
    col="Metric",
    kind="bar",
    sharey=False,
    height=4,
    aspect=1.05,
    palette="viridis"
)

g.set_axis_labels("", "Value")
g.set_titles("{col_name}")

for ax in g.axes.flat:
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    # Add value labels above bars.
    # This version is compatible with older matplotlib versions
    # where ax.bar_label() does not exist.
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()

            if pd.notna(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

g.fig.suptitle(
    "Comparison of model performance on the test set",
    y=1.08,
    fontsize=16,
    fontweight="bold"
)

g.fig.tight_layout()
g.fig.savefig(FIG_DIR / "model_performance_comparison.pdf", bbox_inches="tight")
g.fig.savefig(FIG_DIR / "model_performance_comparison.png", bbox_inches="tight")
plt.show()
