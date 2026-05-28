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
import rasterio
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

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

dst_crs = "EPSG:32643" 

BANDS = ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']

#INDEX FOR 10 BAND ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']]
BAND_10 = [6, 4, 2, 10, 8]

#INDEX FOR 5 BAND ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']]
BAND_5 = [3, 2, 1, 5, 4]


for uav_date in uav_dates:
    uav_raster = f'E:/2023-24 SyedanWalan Season/{uav_date}/{uav_date} Ortho.tif'

    uav_data = []
    with rasterio.open(uav_raster) as uav:
        if uav.count == 10:
            band_indices = BAND_10
        elif uav.count == 5:
            band_indices = BAND_5

        print(uav.read(1).shape)
        for uav_band in band_indices:
            u_band = uav.read(uav_band)
            u_band_flat = u_band.flatten()
            uav_data.append(u_band_flat)

        kwargs = uav.meta.copy()
        kwargs.update({
            "crs": uav.crs,
            "transform": uav.transform,
            "width": uav.width,
            "height": uav.height,
            "count": 1,
            "dtype": rasterio.float32
        })

    uav_data = np.array(uav_data).T
    print(uav_data.shape)

    for band_name in sentinel_band_names:
        rf_model = joblib.load(f"Model Params/{uav_date} {band_name}_Random_Forest_Model.joblib")
        print(f"Loaded models for {uav_date} {band_name}")



        #mask = np.isnan(uav_data).any(axis=1)
        print("Predicting...")
        rf_pred = rf_model.predict(uav_data).astype(np.float32)
        print("Prediction done.")

        print(np.unique(rf_pred))
        #rf_pred[mask] = np.nan

        print("Reshaping...")
        rf_pred_reshaped = rf_pred.reshape((u_band.shape[0], u_band.shape[1])).astype(np.float32)
        print("Reshaping done.")

        print("Saving...")

        path = f"Model Predictions/NATIVE {uav_date} {band_name}_RF_Prediction.tif"
        with rasterio.open(path, 'w', **kwargs) as dst:
            dst.write(rf_pred_reshaped.astype(rasterio.float32), 1)
        print(rf_pred_reshaped.shape)
        print("Saving done.")