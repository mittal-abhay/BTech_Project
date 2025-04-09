import requests
import pandas as pd


def prepare_data (start, end, latitude, longitude):
    # NASA POWER API endpoint for hourly data with specified parameters
    parameters = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS2M,ALLSKY_SFC_SW_DIFF,PS"
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters={parameters}&start={start}&end={end}"
        f"&latitude={latitude}&longitude={longitude}&community=AG&format=JSON"
    )

    # Fetch the data
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Extract the parameters (features)
        features = json_data.get("properties", {}).get("parameter", {})
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(features)

        # The index contains the timestamps; convert it to datetime format
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        # Set the index to the datetime column
        df.index.name = "date_time"

        # Rename columns to match your feature names
        df.rename(
            columns={
                "ALLSKY_SFC_SW_DWN": "nwp_globalirrad",
                "CLRSKY_SFC_SW_DWN": "nwp_directirrad",
                "T2M": "nwp_temperature",
                "RH2M": "nwp_humidity",
                "WS2M": "nwp_windspeed",
                "ALLSKY_SFC_SW_DIFF": "lmd_diffuseirrad",
                "PS": "lmd_pressure",
            },
            inplace=True,
        )

        # Add derived features for `lmd_totalirrad` and `lmd_temperature` (for now they are same as nwp, can be changed later)
        df["lmd_totalirrad"] = df["nwp_globalirrad"]
        df["lmd_temperature"] = df["nwp_temperature"]

        # Resample the data to 30-minute intervals
        df_30min = df.resample("30T").interpolate()

        # Round off any decimal values to 4 decimal places
        df_30min = df_30min.round(4)

        print("Dataset created successfully.")
    else:
        print(f"Failed to fetch data from NASA POWER API. Error code: {response.status_code}. {response.text}")

    # return the Dataframe for further processing
    return df_30min

