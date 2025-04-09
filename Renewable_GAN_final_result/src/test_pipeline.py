from data_init import prepare_data

# Define coordinates for Singapore (port area)
latitude = 1.2644
longitude = 103.8233

# Define the time period
start = "20210101"  # Format: YYYYMMDD
end = "20211231"    # Format: YYYYMMDD


# Prepare the data
weather_df = prepare_data(start, end, latitude, longitude)


# Feed into pv_model and add predicted_pv column to the dataframe

# Feed into load_model and add predicted_load column to the dataframe

# Feed into usep_model and add predicted_usep column to the dataframe

# find cost and add cost column to the dataframe