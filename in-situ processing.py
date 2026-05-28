import pandas as pd


import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from pyproj import Transformer
import numpy as np

# -----------------------------------
# Inputs
# -----------------------------------


gps_coords = [
    [74.1657529999999, 31.07851],           #plot 11_0
    [74.1670069999999, 31.0785039999999],   #plot 11_1
    [74.1669729999999, 31.0795679999999],   #plot 11_2
    [74.165727, 31.0795759999999],          #plot 11_3
    [74.166425, 31.0789809999999]]  


okara_lat = 30.8  # approximate centre of Okara
radius_deg = 10 / (111320 * np.cos(np.radians(okara_lat)))

# = 5 / (111320 * 0.8590)
# = 5 / 95623
# ≈ 0.0000523 degrees

uav_dates = ['2023-12-08',
         '2023-12-16',
         '2023-12-24',
         '2024-01-25',
         '2024-02-02',
         '2024-02-18',
         '2024-03-05',
         '2024-03-13',
         '2024-03-21',
         '2024-03-29',
         '2024-04-22']

manual_dict = {'Dates': uav_dates}
auto_dict = {'Dates': uav_dates}

for i, (lon, lat) in enumerate(gps_coords):
    manual_dict[f"plot_{i}"] = []
    auto_dict[f"plot_{i}"] = []

    for uav_date in uav_dates:
        manual_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Manual SM.tif"
        auto_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Auto SM.tif"

        with rasterio.open(manual_uav_path) as src:

            raster_crs = src.crs


            circle = Point(lon, lat).buffer(radius_deg)

            out_image, _ = mask(src, [mapping(circle)], crop=True, nodata=np.nan, filled=True)
            data = out_image[0]
            valid = data[~np.isnan(data)]
            mean_val = float(np.mean(valid)) if valid.size else float("nan")

            manual_dict[f"plot_{i}"].append(mean_val)

        with rasterio.open(auto_uav_path) as src:

            raster_crs = src.crs
            circle = Point(lon, lat).buffer(radius_deg)

            out_image, _ = mask(src, [mapping(circle)], crop=True, nodata=np.nan, filled=True)
            data = out_image[0]
            valid = data[~np.isnan(data)]
            mean_val = float(np.mean(valid)) if valid.size else float("nan")

            auto_dict[f"plot_{i}"].append(mean_val)

    
manual_df = pd.DataFrame(manual_dict)
auto_df = pd.DataFrame(auto_dict)

manual_df.to_excel("in-situ sensor data/manual.xlsx", index=False)
auto_df.to_excel("in-situ sensor data/auto.xlsx", index=False)


