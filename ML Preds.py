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

# Red, NIR, SWIR2
sentinel_bands = [0,3,5]

sentinel_band_names = ['Red', 'NIR', 'SWIR2']

# Red, Green, Blue, NIR, Red-Edge
uav_bands = [0,1,2,3,4]

def predictor(resolution):
    for uav_date in uav_dates:
        if resolution == 10:
            uav_path = f"UAV Data/{uav_date} Aligned {resolution}m Cropped.tif"
        else:
            uav_path = f"UAV Data/{uav_date} Aligned {resolution}m.tif"
        uav_data = []
        with rasterio.open(uav_path) as uav:
            for uav_band in uav_bands:
                u_band = uav.read(uav_band + 1)
                u_band_flat = u_band.flatten()
                uav_data.append(u_band_flat)

            kwargs = uav.meta.copy()
            kwargs.update({
                "crs": uav.crs,
                "transform": uav.transform,
                "width": uav.width,
                "height": uav.height,
                "count": 1
            })

        uav_data = np.array(uav_data).T
        for band_name in sentinel_band_names:
            rf_model = joblib.load(f"Model Params/{uav_date} {band_name}_Random_Forest_Model.joblib")
            print(f"Loaded models for {uav_date} {band_name}")

            print(uav_data.shape)

            mask = np.isnan(uav_data).any(axis=1)
            rf_pred = rf_model.predict(uav_data)

            rf_pred[mask] = np.nan
            #print(rf_pred.shape)

            rf_pred_reshaped = rf_pred.reshape((u_band.shape[0], u_band.shape[1]))

            if resolution == 10:
                path = f"Model Predictions/{uav_date} {band_name}_RF_Prediction.tif"
            else:
                path = f"Model Predictions/{uav_date} {band_name}_RF_Prediction_{resolution}m.tif"
            with rasterio.open(path, 'w', **kwargs) as dst:
                dst.write(rf_pred_reshaped.astype(rasterio.float32), 1)
            print(rf_pred_reshaped.shape)

predictor(1)




