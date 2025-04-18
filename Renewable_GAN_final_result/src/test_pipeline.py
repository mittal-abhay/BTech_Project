import pandas as pd
from data_init import prepare_data
from models.pv.test import pv_model
from models.load.train import load_model
from models.usep.train import usep_model


# Define coordinates for Singapore (port area)
latitude = 1.2644
longitude = 103.8233

# Define the time period
start = "20210101"  # Format: YYYYMMDD
end = "20210131"    # Format: YYYYMMDD


# Prepare the data
weather_df = prepare_data(start, end, latitude, longitude)
weather_df.to_csv("prepared_data.csv", index=False)

# # Feed into pv_model and add predicted_pv column to the dataframe
print("Running pv_model...")
data_with_predicted_pv = pv_model("prepared_data.csv")
data_with_predicted_pv.to_csv("prepared_data.csv", index=False)
print("Successfully added predicted_pv column to the dataframe.")


# Feed into load_model and add predicted_load column to the dataframe
print("Running load_model...")
data_with_predicted_load = load_model("prepared_data.csv")
data_with_predicted_load.to_csv("prepared_data.csv", index=False)
print("Successfully added predicted_load column to the dataframe.")
# Feed into usep_model and add predicted_usep column to the dataframe

print("Running usep model...")
data_with_predicted_usep = usep_model("prepared_data.csv")
data_with_predicted_usep.to_csv("prepared_data.csv", index=False)
print("Successfully added predicted_usep column to the dataframe.")

# find cost and add predicted_cost column to the dataframe
data_with_predicted_usep['predicted_cost'] = data_with_predicted_usep['predicted_USEP'] * data_with_predicted_usep[['predicted_load', 'predicted_pv']].apply(lambda x: max(0, x[0] - x[1]), axis=1)
data_with_predicted_usep.to_csv("prepared_data.csv", index=False)
print("Successfully added predicted_cost column to the dataframe.")
