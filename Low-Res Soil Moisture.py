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
from scipy.stats import gaussian_kde

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




def ubrmse(real,pred):
    real = np.array(real)
    pred = np.array(pred)
    real_mean = np.mean(real)
    pred_mean = np.mean(pred)
    bias = pred_mean - real_mean
    unb_pred = pred - bias
    ubrmse = np.sqrt(np.mean((real - unb_pred)**2))
    return round(ubrmse, 4)

def get_metrics(true_values, predicted_values):
    mask1 = ~np.isnan(true_values)
    mask2 = ~np.isnan(predicted_values)

    mask = mask1 & mask2

    true_values = np.array(true_values)[mask]
    predicted_values = np.array(predicted_values)[mask]
    mae = mean_absolute_error(true_values, predicted_values)
    #rmse = math.sqrt(mean_squared_error(true_values, predicted_values))
    r, _ = pearsonr(true_values, predicted_values)
    #sp_r, _ = spearmanr(true_values, predicted_values)
    unbiaised_rmse = ubrmse(true_values, predicted_values)
    #r2_score_value = r2_score(true_values, predicted_values)
    return round(mae, 4), round(r, 4), unbiaised_rmse

theta_d = 0.15
theta_w = 0.36

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


manual_uav_all = []
manual_sentinel_all = []
auto_uav_all = []
auto_sentinel_all = []

for i,uav_date in enumerate(uav_dates):
    sentinel_date = sentinel_dates[i]
    sentinel_file = f"Sentinel Data/{sentinel_date} SENTINEL Cropped.tif"

    with rasterio.open(sentinel_file) as sentinel:
        sentinel_red = sentinel.read(sentinel_bands[0]+1)
        sentinel_nir = sentinel.read(sentinel_bands[1]+1)
        sentinel_swir2 = sentinel.read(sentinel_bands[2]+1)
        sentinel_meta = sentinel.meta.copy()

    sentinel_ndvi = (sentinel_nir - sentinel_red)/(sentinel_nir + sentinel_red)
    sentinel_str = (1-sentinel_swir2)**2/(2*sentinel_swir2)

    manual_wet_str = np.polyval([manual_w_m[i],manual_w_c[i]], sentinel_ndvi)
    manual_dry_str = np.polyval([manual_d_m[i],manual_d_c[i]], sentinel_ndvi)

    auto_wet_str = np.polyval([auto_w_m[i],auto_w_c[i]], sentinel_ndvi)
    auto_dry_str = np.polyval([auto_d_m[i],auto_d_c[i]], sentinel_ndvi)

    manual_sentinel_SM = theta_d + (theta_w - theta_d)*(manual_dry_str - sentinel_str)/(manual_dry_str - manual_wet_str)
    manual_sentinel_SM = np.clip(manual_sentinel_SM, theta_d, theta_w)

    auto_sentinel_SM = theta_d + (theta_w - theta_d)*(auto_dry_str - sentinel_str)/(auto_dry_str - auto_wet_str)
    auto_sentinel_SM = np.clip(auto_sentinel_SM, theta_d, theta_w)


    uav_preds = []
    uav_preds_1m = []
    for band_name in sentinel_band_names:
        path = f"Model Predictions/{uav_date} {band_name}_RF_Prediction.tif"
        with rasterio.open(path) as uav:
            uav_pred = uav.read(1)
            uav_preds.append(uav_pred)

        path = f"Model Predictions/{uav_date} {band_name}_RF_Prediction_1m.tif"
        with rasterio.open(path) as uav:
            uav_pred = uav.read(1)
            uav_preds_1m.append(uav_pred)

    uav_red = uav_preds[0]
    uav_nir = uav_preds[1]
    uav_swir2 = uav_preds[2]

    uav_red_1m = uav_preds_1m[0]
    uav_nir_1m = uav_preds_1m[1]
    uav_swir2_1m = uav_preds_1m[2]

    uav_ndvi = (uav_nir - uav_red)/(uav_nir + uav_red)
    uav_str = (1-uav_swir2)**2/(2*uav_swir2)

    uav_ndvi_1m = (uav_nir_1m - uav_red_1m)/(uav_nir_1m + uav_red_1m)
    uav_str_1m = (1-uav_swir2_1m)**2/(2*uav_swir2_1m)

    manual_wet_str_uav = np.polyval([manual_w_m[i],manual_w_c[i]], uav_ndvi)
    manual_dry_str_uav = np.polyval([manual_d_m[i],manual_d_c[i]], uav_ndvi)

    auto_wet_str_uav = np.polyval([auto_w_m[i],auto_w_c[i]], uav_ndvi)
    auto_dry_str_uav = np.polyval([auto_d_m[i],auto_d_c[i]], uav_ndvi)

    manual_wet_str_uav_1m = np.polyval([manual_w_m[i],manual_w_c[i]], uav_ndvi_1m)
    manual_dry_str_uav_1m = np.polyval([manual_d_m[i],manual_d_c[i]], uav_ndvi_1m)

    auto_wet_str_uav_1m = np.polyval([auto_w_m[i],auto_w_c[i]], uav_ndvi_1m)
    auto_dry_str_uav_1m = np.polyval([auto_d_m[i],auto_d_c[i]], uav_ndvi_1m)


    manual_uav_SM = theta_d + (theta_w - theta_d)*(manual_dry_str_uav - uav_str)/(manual_dry_str_uav - manual_wet_str_uav)
    manual_uav_SM = np.clip(manual_uav_SM, theta_d, theta_w)

    auto_uav_SM = theta_d + (theta_w - theta_d)*(auto_dry_str_uav - uav_str)/(auto_dry_str_uav - auto_wet_str_uav)
    auto_uav_SM = np.clip(auto_uav_SM, theta_d, theta_w)

    manual_uav_SM_1m = theta_d + (theta_w - theta_d)*(manual_dry_str_uav_1m - uav_str_1m)/(manual_dry_str_uav_1m - manual_wet_str_uav_1m)
    manual_uav_SM_1m = np.clip(manual_uav_SM_1m, theta_d, theta_w)

    auto_uav_SM_1m = theta_d + (theta_w - theta_d)*(auto_dry_str_uav_1m - uav_str_1m)/(auto_dry_str_uav_1m - auto_wet_str_uav_1m)
    auto_uav_SM_1m = np.clip(auto_uav_SM_1m, theta_d, theta_w)


    manual_sent_path = f"Soil Moisture Predictions/{sentinel_date} SENTINEL Manual SM.tif"
    auto_sent_path = f"Soil Moisture Predictions/{sentinel_date} SENTINEL Auto SM.tif"
    manual_uav_path = f"Soil Moisture Predictions/{uav_date} UAV Manual SM.tif"
    auto_uav_path = f"Soil Moisture Predictions/{uav_date} UAV Auto SM.tif"

    sent_kwargs = sentinel.meta.copy()
    sent_kwargs.update({
        "crs": sentinel.crs,
        "transform": sentinel.transform,
        "width": sentinel.width,
        "height": sentinel.height,
        "count": 1  # all 5 bands in one file
    })

    uav_kwargs = uav.meta.copy()
    uav_kwargs.update({
        "crs": uav.crs,
        "transform": uav.transform,
        "width": uav.width,
        "height": uav.height,
        "count": 1  # all 5 bands in one file
    })
    with rasterio.open(manual_sent_path, 'w', **sent_kwargs) as dst:
        dst.write(manual_sentinel_SM.astype(rasterio.float32), 1)
    with rasterio.open(auto_sent_path, 'w', **sent_kwargs) as dst:
        dst.write(auto_sentinel_SM.astype(rasterio.float32), 1)
    with rasterio.open(manual_uav_path, 'w', **uav_kwargs) as dst:
        dst.write(manual_uav_SM.astype(rasterio.float32), 1)
    with rasterio.open(auto_uav_path, 'w', **uav_kwargs) as dst:
        dst.write(auto_uav_SM.astype(rasterio.float32), 1)

    sentinel_manual_valid = ~np.isnan(manual_sentinel_SM)
    sentinel_auto_valid = ~np.isnan(auto_sentinel_SM)
    uav_manual_valid = ~np.isnan(manual_uav_SM)
    uav_auto_valid = ~np.isnan(auto_uav_SM)

    manual_valid_mask = sentinel_manual_valid & uav_manual_valid
    auto_valid_mask = sentinel_auto_valid & uav_auto_valid

    sentinel_manual_common = np.where(manual_valid_mask, manual_sentinel_SM, np.nan)
    uav_manual_common = np.where(manual_valid_mask, manual_uav_SM, np.nan)
    sentinel_auto_common = np.where(auto_valid_mask, auto_sentinel_SM, np.nan)
    uav_auto_common = np.where(auto_valid_mask, auto_uav_SM, np.nan)

    manual_mask1 = ~np.isnan(uav_manual_common)
    manual_mask2 = ~np.isnan(sentinel_manual_common)
    manual_mask = manual_mask1 & manual_mask2

    manual_sentinel_values = np.array(sentinel_manual_common)[manual_mask]
    manual_uav_values = np.array(uav_manual_common)[manual_mask]

    manual_sentinel_all.extend(manual_sentinel_values)
    manual_uav_all.extend(manual_uav_values)

    auto_mask1 = ~np.isnan(uav_auto_common)
    auto_mask2 = ~np.isnan(sentinel_auto_common)
    auto_mask = auto_mask1 & auto_mask2

    auto_sentinel_values = np.array(sentinel_auto_common)[auto_mask]
    auto_uav_values = np.array(uav_auto_common)[auto_mask]

    auto_sentinel_all.extend(auto_sentinel_values)
    auto_uav_all.extend(auto_uav_values)

    manual_mae, manual_r, manual_ubrmse = get_metrics(sentinel_manual_common, uav_manual_common)
    auto_mae, auto_r, auto_ubrmse = get_metrics(sentinel_auto_common, uav_auto_common)

    print(f"{uav_date} - Manual: MAE={manual_mae}, R={manual_r}, UBRMSE={manual_ubrmse} | Auto: MAE={auto_mae}, R={auto_r}, UBRMSE={auto_ubrmse}")
    print("-"*50)
    
    vmin = theta_d
    vmax = theta_w
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40,10))
    #fig.suptitle(f'{uav_date}')
    im1 = ax1.imshow(auto_sentinel_SM, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    ax1.set_title(f"{uav_date} Sentinel", fontsize= 40)
    im2 = ax2.imshow(auto_uav_SM, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    ax2.set_title(f"{uav_date} UAV 10m", fontsize= 40)
    im3 = ax3.imshow(auto_uav_SM_1m, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
    ax3.set_title(f"{uav_date} UAV Native Resolution", fontsize= 40)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.grid(False)
    cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3])
    cbar.ax.tick_params(labelsize=40)
    fig.savefig(f"trapezoids/{uav_date} comparison.png")

    #plt.show()
    plt.close()

manual_mae, manual_r, manual_ubrmse = get_metrics(manual_sentinel_all, manual_uav_all)
auto_mae, auto_r, auto_ubrmse = get_metrics(auto_sentinel_all, auto_uav_all)

print(len(manual_sentinel_all))
'''x, y = np.array(manual_sentinel_all).flatten(), np.array(manual_uav_all).flatten()

print(x)
print(y)
xy = np.vstack([x, y])
density = gaussian_kde(xy)(xy)

idx = density.argsort()
x, y, density = x[idx], y[idx], density[idx]

fig, ax = plt.subplots()
im1 = ax.scatter(manual_sentinel_all, manual_uav_all, c=density, cmap='gist_rainbow_r', s=1, alpha=0.5)
ax.plot([vmin, vmax], [vmin, vmax], color='red', linewidth=1, linestyle='--', label='1:1')
ax.set_aspect('equal')
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_xlabel("Sentinel Predicted SM")
ax.set_ylabel("UAV Predicted SM")
ax.set_title("UAV vs Sentinel SM (MANUAL EDGES)")
ax.text(0.2,0.325,f"MAE={manual_mae}\nR={manual_r}\nUBRMSE={manual_ubrmse}")
fig.colorbar(im1, ax=[ax])
fig.savefig("trapezoids/Manual Scatter.png")
plt.close()

fig, ax = plt.subplots()
im1 = ax.scatter(auto_sentinel_all, auto_uav_all, c=density, cmap='gist_rainbow_r', s=1, alpha=0.5)
ax.plot([vmin, vmax], [vmin, vmax], color='red', linewidth=1, linestyle='--', label='1:1')
ax.set_aspect('equal')
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_xlabel("Sentinel Predicted SM")
ax.set_ylabel("UAV Predicted SM")
ax.set_title("UAV vs Sentinel SM (AUTO EDGES)")
ax.text(0.2,0.325,f"MAE={auto_mae}\nR={auto_r}\nUBRMSE={auto_ubrmse}")
fig.colorbar(im1, ax=[ax])
fig.savefig("trapezoids/Auto Scatter.png")
plt.close()
'''

print(f"OVERALL - Manual: MAE={manual_mae}, R={manual_r}, UBRMSE={manual_ubrmse} | Auto: MAE={auto_mae}, R={auto_r}, UBRMSE={auto_ubrmse}")