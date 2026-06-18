# 4. Cleaning missing values and creating a datetime variable

df = raw_df.copy()

# Replace the dataset-specific missing value code by NaN
numeric_candidates = [col for col in df.columns if col not in ["Date", "Time"]]
df[numeric_candidates] = df[numeric_candidates].replace(-200, np.nan)

# Combine Date and Time into a single datetime column
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

df = df.sort_values("Datetime").reset_index(drop=True)

missing_before = df.isna().sum().sort_values(ascending=False)
print("Missing values before imputation:")
display(missing_before[missing_before > 0].to_frame("missing_count"))

# Remove NMHC(GT), which is mostly missing
df = df.drop(columns=["NMHC(GT)"], errors="ignore")

# Target variable
TARGET = "CO(GT)"

# Remove rows without target values
df = df.dropna(subset=[TARGET]).copy()

# Interpolate remaining numeric missing values in chronological order
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

missing_after = df.isna().sum().sort_values(ascending=False)

print(f"Shape after cleaning: {df.shape}")
print("\nRemaining missing values:")
display(missing_after[missing_after > 0].to_frame("missing_count"))

display(df.head())
