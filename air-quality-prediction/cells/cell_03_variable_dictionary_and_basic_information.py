# ============================================================
# 3. Variable dictionary and basic information
# ============================================================

variable_descriptions = {
    "Date": "Measurement date",
    "Time": "Measurement time",
    "CO(GT)": "Reference carbon monoxide concentration",
    "PT08.S1(CO)": "Sensor 1 response, mainly sensitive to CO",
    "NMHC(GT)": "Non-methane hydrocarbons",
    "C6H6(GT)": "Reference benzene concentration",
    "PT08.S2(NMHC)": "Sensor 2 response, mainly sensitive to NMHC",
    "NOx(GT)": "Reference nitrogen oxides concentration",
    "PT08.S3(NOx)": "Sensor 3 response, mainly sensitive to NOx",
    "NO2(GT)": "Reference nitrogen dioxide concentration",
    "PT08.S4(NO2)": "Sensor 4 response, mainly sensitive to NO2",
    "PT08.S5(O3)": "Sensor 5 response, mainly sensitive to ozone",
    "T": "Temperature",
    "RH": "Relative humidity",
    "AH": "Absolute humidity",
}

summary_info = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": [raw_df[col].dtype for col in raw_df.columns],
    "unique_values": [raw_df[col].nunique(dropna=True) for col in raw_df.columns],
    "description": [variable_descriptions.get(col, "No description available") for col in raw_df.columns],
})

display(summary_info)
