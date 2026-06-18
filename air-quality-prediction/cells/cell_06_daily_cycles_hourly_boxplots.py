# 6. Daily cycles: hourly boxplots

variables_to_plot = [
    "CO(GT)", "PT08.S1(CO)", "C6H6(GT)",
    "NOx(GT)", "NO2(GT)", "T", "RH", "AH"
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
axes = axes.ravel()

for ax, var in zip(axes, variables_to_plot):
    sns.boxplot(
        data=df,
        x="Hour",
        y=var,
        ax=ax,
        color=sns.color_palette("viridis", 8)[3],
        fliersize=0.6,
        linewidth=0.8
    )
    ax.set_title(var)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00", "06", "12", "18", "23"])

fig.suptitle("Hourly distribution of pollutants and meteorological variables", y=1.03, fontsize=16, fontweight="bold")
save_figure("hourly_cycles_boxplots")
plt.show()
