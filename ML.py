import rasterio
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib


def rf_model(UAV,SENTINEL_BANDS,type):
    print(f"Training {type} Model...")
    uav_train = UAV.T
    sent_train = SENTINEL_BANDS.T.ravel()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(uav_train, sent_train)
    rf_preds = rf_model.predict(uav_train)
    rf_test_r = stats.pearsonr(rf_preds, sent_train)[0]
    print(f"Random Forest R: {rf_test_r:.4f}")
    rf_test_mae = np.mean(np.abs(rf_preds - sent_train))
    print(f"Random Forest MAE: {rf_test_mae:.4f}")

    print(sent_train[:5])
    print(rf_preds[:5])
    return rf_model

def lin_reg_model(UAV,SENTINEL_BANDS,type):
    print(f"Training {type} Model...")
    uav_train = UAV.T
    sent_train = SENTINEL_BANDS.T.ravel()
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(uav_train, sent_train)
    lin_reg_preds = lin_reg_model.predict(uav_train)
    lin_reg_test_r = stats.pearsonr(lin_reg_preds, sent_train)[0]
    print(f"Linear Regression R: {lin_reg_test_r:.4f}")
    lin_reg_test_mae = np.mean(np.abs(lin_reg_preds - sent_train))
    print(f"Linear Regression MAE: {lin_reg_test_mae:.4f}")
    return lin_reg_model



def svr_model(UAV,SENTINEL_BANDS,type):
    print(f"Training {type} Model...")
    uav_train = UAV.T
    sent_train = SENTINEL_BANDS.T.ravel()
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr_model.fit(uav_train, sent_train)
    svr_preds = svr_model.predict(uav_train)
    svr_test_r = stats.pearsonr(svr_preds, sent_train)[0]
    svr_test_mae = np.mean(np.abs(svr_preds - sent_train))

    print(f"SVR R: {svr_test_r:.4f}")
    print(f"SVR MAE: {svr_test_mae:.4f}")

    return svr_model

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

sentinel_dates = ['2023-12-08',
              '2023-12-18',
              '2023-12-21',
              '2024-01-27',
              '2024-02-01',
              '2024-02-16',
              '2024-03-07',
              '2024-03-12',
              '2024-03-22',
              '2024-04-01',
              '2024-04-21']


# Red, NIR, SWIR2
sentinel_bands = [0,3,5]

sentinel_band_names = ['Red', 'NIR', 'SWIR2']

# Red, Green, Blue, NIR, Red-Edge
uav_bands = [0,1,2,3,4]


red_sentinel_data = []
nir_sentinel_data = []
swir2_sentinel_data = []

uav_data = []

for i,uav_date in enumerate(uav_dates):
    print(f"Processing {uav_date}...")
    sentinel_date = sentinel_dates[i]

    uav_path = f"UAV Data/{uav_date} Aligned 10m Cropped.tif"
    sentinel_out = f"Sentinel Data/{sentinel_date} SENTINEL Cropped.tif"

    uav_data = []
    sentinel_data = []

    with rasterio.open(uav_path) as uav, rasterio.open(sentinel_out) as sentinel:
        for uav_band in uav_bands:
            u_band = uav.read(uav_band + 1)
            u_band_flat = u_band.flatten()
            uav_data.append(u_band_flat)

        for j, sentinel_band in enumerate(sentinel_bands):
            s_band = sentinel.read(sentinel_band + 1)
            s_band_flat = s_band.flatten()
            sentinel_data.append(s_band_flat)

    uav_data = np.array(uav_data).T
    sentinel_data = np.array(sentinel_data).T

    uav_nan_index = np.isnan(uav_data).any(axis=1)
    sentinel_nan_index = np.isnan(sentinel_data).any(axis=1)

    combined_nan_index = uav_nan_index | sentinel_nan_index

    filtered_uav_data = uav_data[~combined_nan_index].T
    filtered_sentinel_data = sentinel_data[~combined_nan_index].T


    for j, band_name in enumerate(sentinel_band_names):
        uav_band_data = filtered_uav_data
        sentinel_band_data = np.array([filtered_sentinel_data[j]])

        svr = svr_model(uav_band_data, sentinel_band_data, f"{band_name} SVR")
        lin_reg = lin_reg_model(uav_band_data, sentinel_band_data, f"{band_name} Linear Regression")
        rf = rf_model(uav_band_data, sentinel_band_data, f"{band_name} Random Forest")

        joblib.dump(svr, f"Model Params/{uav_date} {band_name}_SVR_Model.joblib")
        joblib.dump(lin_reg, f"Model Params/{uav_date} {band_name}_Linear_Regression_Model.joblib")
        joblib.dump(rf, f"Model Params/{uav_date} {band_name}_Random_Forest_Model.joblib")
    print("-" * 50)


