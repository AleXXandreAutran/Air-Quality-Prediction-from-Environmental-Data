# ============================================================
# 24. Final recommendation
# ============================================================

best_model_name = results.iloc[0]["Model"]

print(f"Best predictive model according to RMSE: {best_model_name}")
print("\nInterpretation:")
print(
    "- KNN is recommended for pure prediction because it achieves the lowest error.\n"
    "- Ridge is recommended for interpretation because it handles multicollinearity "
    "while keeping readable coefficients.\n"
    "- Lasso does not perform strong variable selection here because the optimal alpha is very small."
)
