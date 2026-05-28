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
    rf_test_urmse = ubrmse(rf_preds, sent_train)
    print(f"Random Forest uRMSE: {rf_test_urmse:.4f}")


    return rf_model,rf_preds

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
    lin_reg_test_urmse = ubrmse(lin_reg_preds, sent_train)
    print(f"Linear Regression uRMSE: {lin_reg_test_urmse:.4f}")
    return lin_reg_model, lin_reg_preds



def svr_model(UAV,SENTINEL_BANDS,type):
    print(f"Training {type} Model...")
    uav_train = UAV.T
    sent_train = SENTINEL_BANDS.T.ravel()
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr_model.fit(uav_train, sent_train)
    svr_preds = svr_model.predict(uav_train)
    svr_test_r = stats.pearsonr(svr_preds, sent_train)[0]
    svr_test_mae = np.mean(np.abs(svr_preds - sent_train))
    svr_test_urmse = ubrmse(svr_preds, sent_train)
    print(f"SVR R: {svr_test_r:.4f}")
    print(f"SVR MAE: {svr_test_mae:.4f}")
    print(f"SVR uRMSE: {svr_test_urmse:.4f}")


    return svr_model, svr_preds

def ubrmse(real,pred):
    real = np.array(real)
    pred = np.array(pred)
    real_mean = np.mean(real)
    pred_mean = np.mean(pred)
    bias = pred_mean - real_mean
    unb_pred = pred - bias
    ubrmse = np.sqrt(np.mean((real - unb_pred)**2))
    return round(ubrmse, 4)

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
def top_n(uav_dates, sentinel_dates, n):

    all_real = {band: [] for band in sentinel_band_names}
    all_svr_pred = {band: [] for band in sentinel_band_names}
    all_lin_reg_pred = {band: [] for band in sentinel_band_names}
    all_rf_pred = {band: [] for band in sentinel_band_names}

    for i,uav_date in enumerate(uav_dates):
        print(f"Processing {uav_date}...")
        sentinel_date = sentinel_dates[i]

        uav_path = f"UAV Data/{uav_date} Aligned 10m Cropped.tif"
        sentinel_path = f"Sentinel Data/{sentinel_date} SENTINEL Cropped.tif"




        with rasterio.open(sentinel_path) as sentinel:
            for sentinel_band_type, sentinel_band_idx in zip(sentinel_band_names, sentinel_bands):
                uav_data = []
                sentinel_data = []
                sentinel_band = sentinel.read(sentinel_band_idx+1)

                sentinel_band_flat = sentinel_band.flatten()
                sentinel_data.append(sentinel_band_flat)

                if n==5:
                    top_indices = uav_bands

                else:
                    ranked_indices = np.load(f"Ranked Indices/{sentinel_date} {sentinel_band_type} Sorted Indices.npy")
                    top_indices = ranked_indices[:n]

                with rasterio.open(uav_path) as uav:
                    for uav_band in top_indices:
                        #print(int(uav_band))
                        u_band = uav.read(int(uav_band) + 1)
                        u_band_flat = u_band.flatten()
                        uav_data.append(u_band_flat)



                uav_data = np.array(uav_data).T
                sentinel_data = np.array(sentinel_data).T

                uav_nan_index = np.isnan(uav_data).any(axis=1)
                sentinel_nan_index = np.isnan(sentinel_data).any(axis=1)

                combined_nan_index = uav_nan_index | sentinel_nan_index

                filtered_uav_data = uav_data[~combined_nan_index].T
                filtered_sentinel_data = sentinel_data[~combined_nan_index].T


                uav_band_data = filtered_uav_data
                sentinel_band_data = np.array([filtered_sentinel_data])

                all_real[sentinel_band_type].extend(sentinel_band_data.T.ravel())

                svr, svr_preds = svr_model(uav_band_data, sentinel_band_data, f"{sentinel_date} {sentinel_band_type} Top {n} SVR")
                lin_reg, lin_reg_preds = lin_reg_model(uav_band_data, sentinel_band_data, f"{sentinel_date} {sentinel_band_type} Top {n} Linear Regression")
                rf, rf_preds = rf_model(uav_band_data, sentinel_band_data, f"{sentinel_date} {sentinel_band_type} Top {n} Random Forest")

                all_svr_pred[sentinel_band_type].extend(svr_preds)
                all_lin_reg_pred[sentinel_band_type].extend(lin_reg_preds)
                all_rf_pred[sentinel_band_type].extend(rf_preds)

                print("-" * 50)

    for sentinel_band_type in sentinel_band_names:  
        svr_r = stats.pearsonr(all_svr_pred[sentinel_band_type], all_real[sentinel_band_type])[0]
        svr_mae = np.mean(np.abs(np.array(all_svr_pred[sentinel_band_type]) - np.array(all_real[sentinel_band_type])))
        svr_ubrmse = ubrmse(all_svr_pred[sentinel_band_type], all_real[sentinel_band_type])

        lin_reg_r = stats.pearsonr(all_lin_reg_pred[sentinel_band_type], all_real[sentinel_band_type])[0]
        lin_reg_mae = np.mean(np.abs(np.array(all_lin_reg_pred[sentinel_band_type]) - np.array(all_real[sentinel_band_type])))
        lin_reg_ubrmse = ubrmse(all_lin_reg_pred[sentinel_band_type], all_real[sentinel_band_type])

        rf_r = stats.pearsonr(all_rf_pred[sentinel_band_type], all_real[sentinel_band_type])[0]
        rf_mae = np.mean(np.abs(np.array(all_rf_pred[sentinel_band_type]) - np.array(all_real[sentinel_band_type])))
        rf_ubrmse = ubrmse(all_rf_pred[sentinel_band_type], all_real[sentinel_band_type])

        print(f"{sentinel_band_type} Top {n}: SVR (R={svr_r:4f}, MAE={svr_mae:4f}, uRMSE={svr_ubrmse:4f}), Linear Regression (R={lin_reg_r:4f}, MAE={lin_reg_mae:4}, uRMSE={lin_reg_ubrmse}), Random Forest (R={rf_r:4}, MAE={rf_mae:4}, uRMSE={rf_ubrmse:4f})")
            #print("-" * 50)


top_n(uav_dates, sentinel_dates, 5)
#top_n(uav_dates, sentinel_dates, 4)
#top_n(uav_dates, sentinel_dates, 3)
#top_n(uav_dates, sentinel_dates, 2)
#top_n(uav_dates, sentinel_dates, 1)