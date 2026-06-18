# 10. Scatterplots of the most correlated variables

top_features = target_corr.head(5).index.tolist()

fig, axes = plt.subplots(1, len(top_features), figsize=(4.2 * len(top_features), 3.8), sharey=True)

for ax, feature in zip(axes, top_features):
    sns.scatterplot(
        data=df.sample(min(len(df), 3000), random_state=RANDOM_STATE),
        x=feature,
        y=TARGET,
        ax=ax,
        alpha=0.35,
        s=18,
        edgecolor=None
    )
    sns.regplot(
        data=df.sample(min(len(df), 3000), random_state=RANDOM_STATE),
        x=feature,
        y=TARGET,
        ax=ax,
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.6}
    )
    ax.set_title(feature)
    ax.set_xlabel(feature)

axes[0].set_ylabel(TARGET)
fig.suptitle("Relationship between CO(GT) and the most correlated predictors", y=1.04, fontsize=15, fontweight="bold")
save_figure("scatter_top_correlations")
plt.show()
