# 5. Create useful time variables for exploratory analysis

df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek  # Monday = 0, Sunday = 6
df["DayName"] = df["Datetime"].dt.day_name()

ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["DayName"] = pd.Categorical(df["DayName"], categories=ordered_days, ordered=True)

display(df[numeric_cols].describe().T.round(3))
