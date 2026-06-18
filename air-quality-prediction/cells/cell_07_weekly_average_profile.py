# 7. Weekly average profile

weekly_means = (
    df.groupby("DayName", observed=True)[variables_to_plot]
    .mean()
    .reset_index()
)

weekly_long = weekly_means.melt(
    id_vars="DayName",
    var_name="Variable",
    value_name="Mean value"
)

g = sns.FacetGrid(
    weekly_long,
    col="Variable",
    col_wrap=4,
    sharey=False,
    height=3.0,
    aspect=1.25
)
g.map_dataframe(
    sns.lineplot,
    x="DayName",
    y="Mean value",
    marker="o",
    linewidth=2
)
g.set_titles("{col_name}")
g.set_axis_labels("", "Mean value")

for ax in g.axes.flat:
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

g.fig.suptitle("Average weekly profile", y=1.04, fontsize=16, fontweight="bold")
g.fig.tight_layout()
g.fig.savefig(FIG_DIR / "weekly_profile.pdf", bbox_inches="tight")
g.fig.savefig(FIG_DIR / "weekly_profile.png", bbox_inches="tight")
plt.show()
