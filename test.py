import ee
import pandas as pd
from datetime import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

def get_monthly_rain_since_last_november(lat, lon):
    # Dates: from last Nov 1 to today
    today = datetime.now()
    year = today.year if today.month >= 11 else today.year - 1
    start_date = datetime(year, 11, 1)

    point = ee.Geometry.Point(lon, lat)

    # ERA5-Land: daily total precipitation
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")) \
        .select("total_precipitation_sum")

    # Extract daily values
    def extract(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        rain = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=1000
        ).get("total_precipitation_sum")
        return ee.Feature(None, {"date": date, "rain": rain})

    fc = ee.FeatureCollection(collection.map(extract))
    dates = fc.aggregate_array("date").getInfo()
    rains = fc.aggregate_array("rain").getInfo()

    # DataFrame: convert and group
    df = pd.DataFrame({"date": pd.to_datetime(dates), "rain": rains})
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce") * 1000  # meters → mm
    df = df.dropna()

    df["month"] = df["date"].dt.to_period("M")
    df_monthly = df.groupby("month")["rain"].sum().reset_index()
    df_monthly["month"] = df_monthly["month"].dt.month

    
    return df_monthly
rain_df = get_monthly_rain_since_last_november(35.27954, -119.169)
print(rain_df)

rain_df['month'].dt.month