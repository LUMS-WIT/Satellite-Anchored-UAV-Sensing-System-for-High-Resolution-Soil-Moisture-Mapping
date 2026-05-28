import rasterio
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from matplotlib.colors import LogNorm


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



theta_d = 0.15
theta_w = 0.36

theta_d = 0.12486
theta_w = 0.43749

theta_d = 0.16085
theta_w = 0.37451



manual_w_m = [12 for _ in range(len(uav_dates))]
manual_w_c = [2  for _ in range(len(uav_dates))]

manual_d_m = [1 for _ in range(len(uav_dates))]
manual_d_c = [0 for _ in range(len(uav_dates))]


auto_w_m = [4.715 for _ in range(len(uav_dates))]
auto_w_c = [5.439  for _ in range(len(uav_dates))]

auto_d_m = [3.182 for _ in range(len(uav_dates))]
auto_d_c = [-0.300 for _ in range(len(uav_dates))]



# Red, NIR, SWIR2
sentinel_bands = [0,3,5]

sentinel_band_names = ['Red', 'NIR', 'SWIR2']

# Red, Green, Blue, NIR, Red-Edge
uav_bands = [0,1,2,3,4]



for i,uav_date in enumerate(uav_dates):
    uav_preds = []
    for band_name in sentinel_band_names:
        path = f"Model Predictions/NATIVE {uav_date} {band_name}_RF_Prediction.tif"
        with rasterio.open(path) as uav:
            uav_pred = uav.read(1)
            uav_preds.append(uav_pred)

    uav_red = uav_preds[0]
    uav_nir = uav_preds[1]
    uav_swir2 = uav_preds[2]

    uav_ndvi = (uav_nir - uav_red)/(uav_nir + uav_red)
    uav_str = (1-uav_swir2)**2/(2*uav_swir2)

    manual_wet_str_uav = np.polyval([manual_w_m[i],manual_w_c[i]], uav_ndvi)
    manual_dry_str_uav = np.polyval([manual_d_m[i],manual_d_c[i]], uav_ndvi)

    auto_wet_str_uav = np.polyval([auto_w_m[i],auto_w_c[i]], uav_ndvi)
    auto_dry_str_uav = np.polyval([auto_d_m[i],auto_d_c[i]], uav_ndvi)

    manual_uav_SM = theta_d + (theta_w - theta_d)*(manual_dry_str_uav - uav_str)/(manual_dry_str_uav - manual_wet_str_uav)
    auto_uav_SM = theta_d + (theta_w - theta_d)*(auto_dry_str_uav - uav_str)/(auto_dry_str_uav - auto_wet_str_uav)

    manual_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Manual SM.tif"
    auto_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Auto SM.tif"


    uav_kwargs = uav.meta.copy()
    uav_kwargs.update({
        "crs": uav.crs,
        "transform": uav.transform,
        "width": uav.width,
        "height": uav.height,
        "count": 1  # all 5 bands in one file
    })

    with rasterio.open(manual_uav_path, 'w', **uav_kwargs) as dst:
        dst.write(manual_uav_SM.astype(rasterio.float32), 1)
    with rasterio.open(auto_uav_path, 'w', **uav_kwargs) as dst:
        dst.write(auto_uav_SM.astype(rasterio.float32), 1)

    print(f"{uav_date} done.")