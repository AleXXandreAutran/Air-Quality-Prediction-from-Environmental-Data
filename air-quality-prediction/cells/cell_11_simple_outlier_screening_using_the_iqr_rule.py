# ============================================================
# 11. Simple outlier screening using the IQR rule
# ============================================================

outlier_summary = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary.append({
        "variable": col,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "potential_outliers": count,
        "percentage": 100 * count / len(df)
    })

outlier_summary = pd.DataFrame(outlier_summary).sort_values("potential_outliers", ascending=False)
display(outlier_summary.round(3))

strong_pairs = []
cols = corr_matrix.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        value = corr_matrix.iloc[i, j]
        if abs(value) > 0.90:
            strong_pairs.append({
                "Variable 1": cols[i],
                "Variable 2": cols[j],
                "Correlation": value
            })

strong_pairs = pd.DataFrame(strong_pairs).sort_values("Correlation", key=lambda s: s.abs(), ascending=False)
display(strong_pairs.round(3))
