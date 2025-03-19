import pandas as pd

# Load the dataset
file_path = "./testing_dataset/sing21.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Convert date_time to datetime format
df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M")

# Extract month, time components
df["month"] = df["date_time"].dt.month
df["time"] = df["date_time"].dt.strftime("%H:%M")
df["hour"] = df["date_time"].dt.hour

# Define night-time hours (assuming 18:00 - 06:00 as night)
night_hours = list(range(18, 24)) + list(range(0, 6))

# Identify rows where power is zero and not in night-time
mask = (df["power"] == 0) & (~df["hour"].isin(night_hours))

# Compute mean power for each time across all days of the same month
mean_power_by_time = df.groupby(["month", "time"])['power'].apply(lambda x: x[x > 0].mean())

# Replace zero power values (only in non-night hours) with corresponding means
df.loc[mask, "power"] = df.loc[mask, ["month", "time"]].apply(
    lambda row: mean_power_by_time.get((row["month"], row["time"]), row["power"]), axis=1
)

# Drop unnecessary columns
df.drop(columns=["month", "time", "hour"], inplace=True)

# Save the updated dataset
updated_file_path = "updated_singapore_weather.csv"  # Update as needed
df.to_csv(updated_file_path, index=False)

print(f"Updated dataset saved to {updated_file_path}")
