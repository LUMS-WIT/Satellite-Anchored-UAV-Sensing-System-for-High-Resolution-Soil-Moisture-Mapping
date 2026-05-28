import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from rasterio.mask import mask
import numpy as np

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

dst_crs = "EPSG:32643" 

BANDS = ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']

#INDEX FOR 10 BAND ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']]
BAND_10 = [6, 4, 2, 10, 8]

#INDEX FOR 5 BAND ['Red', 'Green', 'Blue', 'NIR', 'Red-Edge']]
BAND_5 = [3, 2, 1, 5, 4]

NATIVE_RESOLUTION = 10

def res_channger(resolution):
    for i,uav_date in enumerate(uav_dates): 
        if resolution == NATIVE_RESOLUTION:
            sentinel_date = sentinel_dates[i]      
            sentinel_file = f"Sentinel Data/SyedanWalan_Sentinel_{sentinel_date}.tif"
            uav_raster = f'E:/2023-24 SyedanWalan Season/{uav_date}/{uav_date} Ortho.tif'
            aligned_path = f"UAV Data/{uav_date} Aligned {resolution}m.tif"

            with rasterio.open(uav_raster) as src, rasterio.open(sentinel_file) as ref:
                if src.count == 10:
                    band_indices = BAND_10
                elif src.count == 5:
                    band_indices = BAND_5

                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": ref.crs,
                    "transform": ref.transform,
                    "width": ref.width,
                    "height": ref.height,
                    "count": 5  # all 5 bands in one file
                })

                print(f"Processing {aligned_path}...")
                with rasterio.open(aligned_path, "w", **kwargs) as dst:
                    for out_band, src_band in enumerate(band_indices, start=1):
                        reproject(
                            source=rasterio.band(src, src_band),
                            destination=rasterio.band(dst, out_band),  # 1=Red, 2=Green, etc.
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )
                print(f"Finished {uav_date}")
        else:
            print(resolution)
            sentinel_date = sentinel_dates[i]      
            sentinel_file = f"Sentinel Data/SyedanWalan_Sentinel_{sentinel_date}.tif"
            uav_raster = f'E:/2023-24 SyedanWalan Season/{uav_date}/{uav_date} Ortho.tif'
            aligned_path = f"UAV Data/{uav_date} Aligned {resolution}m.tif"

            with rasterio.open(uav_raster) as src, rasterio.open(sentinel_file) as ref:
                if src.count == 10:
                    band_indices = BAND_10
                elif src.count == 5:
                    band_indices = BAND_5

                transform, width, height = calculate_default_transform(
                src.crs, ref.crs, src.width, src.height, *src.bounds, resolution=resolution
                )

                from rasterio.warp import transform_bounds
                from rasterio.transform import from_bounds
                from rasterio.crs import CRS

                target_crs = CRS.from_epsg(32643)

                dst_bounds = transform_bounds(src.crs, target_crs, *src.bounds)
                dst_left, dst_bottom, dst_right, dst_top = dst_bounds

                print(f"Projected bounds: {dst_bounds}")  # should be large numbers in metres

                width = int((dst_right - dst_left) / resolution)
                height = int((dst_top - dst_bottom) / resolution)

                print(f"Width: {width}, Height: {height}")  # should be thousands x thousands at 1m

                from rasterio.transform import from_bounds
                transform = from_bounds(dst_left, dst_bottom, dst_right, dst_top, width, height)


                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": ref.crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "count": 5
                })

                print(f"Processing {aligned_path}...")
                with rasterio.open(aligned_path, "w", **kwargs) as dst:
                    for out_band, src_band in enumerate(band_indices, start=1):
                        reproject(
                            source=rasterio.band(src, src_band),
                            destination=rasterio.band(dst, out_band),  # 1=Red, 2=Green, etc.
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=ref.crs,
                            resampling=Resampling.bilinear
                        )
                print(f"Finished {uav_date}")

#res_channger(10)
#res_channger(1)
#res_channger(0.08)

dst_crs = "EPSG:32643" 

def resolution_changer(resolution, dates, dst_crs):
    for i, uav_date in enumerate(dates):
        uav_path = f'E:/2023-24 SyedanWalan Season/{uav_date}/{uav_date} Ortho.tif'
        out_path = f"UAV Data/{uav_date} Aligned {resolution}m.tif"

        if uav_date == '2024-03-21' or uav_date == '2024-04-22':
            shapefile_path = f"Shapefile/{uav_date}.shp"
        else:
            shapefile_path = f"Shapefile/Default.shp"

        gdf = gpd.read_file(shapefile_path)

        with rasterio.open(uav_path) as src:
            if src.count == 10:
                band_indices = BAND_10
            elif src.count == 5:
                band_indices = BAND_5

            # Reproject shapefile to match source CRS for masking
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            shapes = [geom.__geo_interface__ for geom in gdf.geometry]

            # Mask and crop to shapefile — this gives us the cropped image and transform
            #out_image, out_transform = mask(src, shapes, crop=True, nodata=65535, filled=True)
            #out_height, out_width = out_image.shape[1], out_image.shape[2]

            out_image, out_transform = mask(src, shapes, crop=True, nodata=0, filled=True)
            out_image = out_image.astype(np.float32)
            out_image[out_image == 0] = np.nan

            out_height, out_width = out_image.shape[1], out_image.shape[2]

            # Now compute the reprojection transform from the CROPPED bounds
            from rasterio.transform import array_bounds
            cropped_bounds = array_bounds(out_height, out_width, out_transform)

            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                out_width, out_height,
                *cropped_bounds,
                resolution=resolution
            )

            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "count": 5,
                "dtype": "float32",  # ← this is the key line you're missing
                "nodata": np.nan
            })

            print(f"Processing {uav_date}: {width}x{height} at {resolution}m")

            with rasterio.open(out_path, "w", **kwargs) as dst:
                for out_band, src_band in enumerate(band_indices, start=1):
                    reproject(
                        source=out_image[src_band - 1],  # use masked array, not src
                        destination=rasterio.band(dst, out_band),
                        src_transform=out_transform,     # use cropped transform
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

            print(f"Finished {uav_date}")

resolution_changer(1, uav_dates, dst_crs)
#resolution_changer(1, dates, dst_crs)

