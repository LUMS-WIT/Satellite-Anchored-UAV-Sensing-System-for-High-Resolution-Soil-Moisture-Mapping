import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def ubrmse(real,pred):
    real = np.array(real)
    pred = np.array(pred)
    real_mean = np.mean(real)
    pred_mean = np.mean(pred)
    bias = pred_mean - real_mean
    unb_pred = pred - bias
    ubrmse = np.sqrt(np.mean((real - unb_pred)**2))
    return ubrmse
uav_dates = [
    '2023-12-08', '2023-12-16', '2023-12-24',
    '2024-01-25', '2024-02-02', '2024-02-18',
    '2024-03-05', '2024-03-13', '2024-03-21',
    '2024-03-29', '2024-04-22',
]

shape_files = ['Default', 'Default', 'Default',
               'Default', 'Default', 'Default',
               'Default', 'Default', '2024-03-21']

records = []

def interval(true_values, predicted_values):
    error = np.abs(np.array(true_values)-np.array(predicted_values))
    num_samples = len(error)
    if num_samples<30:
        s = np.std(error)
        t = 2.365
        mean = np.mean(error)

        error_b = (t*s)/np.sqrt(num_samples)
    else:
        s = np.std(error)
        t = 1.96
        mean = np.mean(error)

        error_b = (t*s)/np.sqrt(num_samples)       

    low = float(mean - error_b)#(t*s)/np.sqrt(num_samples)
    high = float(mean + error_b)

    return round(low,3), round(high,3)

for uav_date in uav_dates:
    manual_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Manual SM.tif"
    auto_uav_path = f"Soil Moisture Predictions/NATIVE {uav_date} UAV Auto SM.tif"

    if uav_date=='2024-03-21' or uav_date=='2024-04-22':
        shapefile_path = f"Shapefile/{uav_date}.shp"
    else:
        shapefile_path = f"Shapefile/Default.shp"

    gdf = gpd.read_file(shapefile_path)

    with rasterio.open(auto_uav_path) as src:
        # reproject shapefile to raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        shapes = [geom.__geo_interface__ for geom in gdf.geometry]
        out_image, _ = mask(src, shapes, crop=True, nodata=np.nan, filled=True)
        data  = out_image[0]
        valid = data[~np.isnan(data)]
        auto_mean_val = float(np.mean(valid)) if valid.size else float("nan")

    with rasterio.open(manual_uav_path) as src:
        # reproject shapefile to raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        shapes = [geom.__geo_interface__ for geom in gdf.geometry]
        out_image, _ = mask(src, shapes, crop=True, nodata=np.nan, filled=True)
        data  = out_image[0]
        valid = data[~np.isnan(data)]
        manual_mean_val = float(np.mean(valid)) if valid.size else float("nan")

    records.append({"Date": uav_date, "AUTO": auto_mean_val, "MANUAL": manual_mean_val})

raster_df = pd.DataFrame(records)
raster_df['Date'] = pd.to_datetime(raster_df['Date'])
#df.to_excel("field_mean_sm.xlsx", index=False)

df = pd.read_excel("SMAP_soil_moisture_data.xlsx", parse_dates=["Date"])
df = df.sort_values("Date")
df["avg"] = df[["SMAP 6AM", "SMAP 6PM"]].mean(axis=1)  
df["6AM_5day"]  = df["SMAP 6AM"].rolling(window=5, min_periods=1).mean()
df["6PM_5day"]  = df["SMAP 6PM"].rolling(window=5, min_periods=1).mean()
df["avg_5day"]  = df["avg"].rolling(window=5, min_periods=1).mean()

merged = raster_df.merge(df[["Date", "6AM_5day", "6PM_5day", "avg_5day"]], on="Date", how="left")

theta_d = 0.16085
theta_w = 0.37451

for uav in ["AUTO", "MANUAL"]:
    for col in ["6AM_5day", "6PM_5day", "avg_5day"]:
        valid_mask = ~np.isnan(merged[uav]) & ~np.isnan(merged[col])
        r, _   = pearsonr(merged.loc[valid_mask, uav], merged.loc[valid_mask, col])
        mae    = np.mean(np.abs(merged.loc[valid_mask, uav] - merged.loc[valid_mask, col]))
        u = ubrmse(merged.loc[valid_mask, uav], merged.loc[valid_mask, col])

        m, b = np.polyfit(merged.loc[valid_mask, col], merged.loc[valid_mask, uav],  1)

        print(interval(merged.loc[valid_mask, uav], merged.loc[valid_mask, col]))
        x_line = np.linspace(0.15, 0.4, 100)
        y_line = m * x_line + b
        plt.scatter(merged.loc[valid_mask, col], merged.loc[valid_mask, uav], s=5)
        plt.plot([0.15, 0.4], [0.15, 0.4], color='red', linewidth=1, linestyle='--', label='1:1')
        plt.plot(x_line, y_line, color='blue', linewidth=1, label=f'Best Fit Line')
        plt.xlabel("SMAP 5 Day Average")
        plt.ylabel("UAV Predicted SM")
        plt.gca().set_aspect('equal')
        plt.xlim(0.15, 0.4)
        plt.ylim(0.15, 0.4)
        plt.text(0.2, 0.3, f"MAE={mae:.4f}\nR={r:.4f}\nUBRMSE={u:.4f}")
        plt.legend()
        plt.savefig(f"trapezoids/{uav} {col} SM comparison.png")
        plt.close()
        print(f"{uav} vs {col}: MAE={mae:.4f}, R={r:.4f}, UBRMSE={u:.4f}")



