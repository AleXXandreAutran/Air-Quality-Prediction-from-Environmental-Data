# ============================================================
# 12. Feature matrix and target vector
# ============================================================

features_to_drop = [TARGET, "Date", "Time", "Datetime", "DayName"]
X = df.drop(columns=features_to_drop, errors="ignore")
y = df[TARGET]

# Keep only numerical features for the models
X = X.select_dtypes(include=[np.number]).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape:     {X_test.shape}")

display(X_train.head())
