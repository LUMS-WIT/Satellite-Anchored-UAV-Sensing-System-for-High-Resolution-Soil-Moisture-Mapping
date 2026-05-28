import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from rasterio.features import shapes


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

BANDS = ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']


def native_aligner(dates):
    for i,uav_date in enumerate(uav_dates): 
        sentinel_date = sentinel_dates[i]      
        #input_paths = [f"UAV Data/{uav_date} {band} {resolution}m.tif" for band in BANDS]
        sentinel_file = f"Sentinel Data/SyedanWalan_Sentinel_{sentinel_date}.tif"
        input_path = f"UAV Data/{uav_date} Aligned 10m.tif"

        with rasterio.open(input_path) as uav, rasterio.open(sentinel_file) as sentinel:
            uav_data = uav.read(1) 
            sentinel_data = sentinel.read(1)

            uav_data_all = uav.read()
            sentinel_data_all = sentinel.read()


            sentinel_valid = ~np.isnan(sentinel_data)
            uav_valid1 = np.where(uav_data < 30000, True, False)
            uav_valid2 = np.where(uav_data > 0, True, False)
            uav_valid = uav_valid1 & uav_valid2

            valid_mask = sentinel_valid & uav_valid

            uav_common = np.where(valid_mask, uav_data_all, np.nan)
            sentinel_common = np.where(valid_mask, sentinel_data_all, np.nan)

            uav_meta = uav.meta.copy()
            uav_meta.update(dtype=rasterio.float32, nodata=np.nan)

            sentinel_meta = sentinel.meta.copy()
            sentinel_meta.update(dtype=rasterio.float32, nodata=np.nan)

            uav_out = f"UAV Data/{uav_date} Aligned 10m Cropped.tif"
            with rasterio.open(uav_out, "w", **uav_meta) as dst:
                dst.write(uav_common.astype(np.float32))

            sentinel_out = f"Sentinel Data/{sentinel_date} SENTINEL Cropped.tif"
            with rasterio.open(sentinel_out, "w", **sentinel_meta) as dst:
                dst.write(sentinel_common.astype(np.float32))

            print(f"Saved {uav_out} and {sentinel_out}")

native_aligner(uav_dates)

'''def all_aligner(resolution):
    for d in dates:

        red_path = d+"/"+d+" Red "+str(resolution)+"m.tif"
        blue_path = d+"/"+d+" Blue "+str(resolution)+"m.tif"
        green_path = d+"/"+d+" Green "+str(resolution)+"m.tif"
        nir_path = d+"/"+d+" NIR "+str(resolution)+"m.tif"
        red_edge_path = d+"/"+d+" Red-Edge "+str(resolution)+"m.tif"

        input_paths = [red_path,
                    blue_path,
                    green_path,
                    nir_path,
                    red_edge_path]
        
        path_types = ['Red',
                    'Blue',
                    'Green',
                    'NIR',
                    'Red-Edge']


        for input_path,path_type in zip(input_paths,path_types):
            with rasterio.open(input_path) as uav:
                uav_data = uav.read(1)  
                uav_data_all = uav.read()

                uav_valid1 = np.where(uav_data < 30000, True, False)
                uav_valid2 = np.where(uav_data > 0, True, False)
                uav_valid = uav_valid1 & uav_valid2

                valid_mask = uav_valid

                uav_common = np.where(valid_mask, uav_data_all, np.nan)

                uav_meta = uav.meta.copy()
                uav_meta.update(dtype=rasterio.float32, nodata=np.nan)

                uav_out = d+"/"+d+" "+path_type+f" Aligned {resolution}m Cropped.tif"
                with rasterio.open(uav_out, "w", **uav_meta) as dst:
                    dst.write(uav_common.astype(np.float32))

all_aligner(0.08)
'''

 