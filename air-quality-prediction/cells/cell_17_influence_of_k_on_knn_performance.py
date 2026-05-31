# ============================================================
# 17. Influence of k on KNN performance
# ============================================================

knn_cv_results = pd.DataFrame(knn_search.cv_results_)
knn_cv_results["RMSE_CV"] = -knn_cv_results["mean_test_score"]

# Select Euclidean + uniform results for a clean k-curve
k_curve = knn_cv_results[
    (knn_cv_results["param_model__weights"] == "uniform") &
    (knn_cv_results["param_model__p"] == 2)
].copy()

k_curve["k"] = k_curve["param_model__n_neighbors"].astype(int)

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=k_curve.sort_values("k"),
    x="k",
    y="RMSE_CV",
    marker="o",
    linewidth=2.2
)
plt.axvline(
    x=knn_search.best_params_["model__n_neighbors"],
    linestyle="--",
    linewidth=1.5,
    color="black",
    label=f"Best k = {knn_search.best_params_['model__n_neighbors']}"
)
plt.xlabel("Number of neighbors k")
plt.ylabel("Cross-validated RMSE")
plt.title("Influence of k on KNN performance")
plt.legend()
save_figure("knn_k_selection")
plt.show()
