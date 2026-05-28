import rasterio
import numpy as np
import scipy.stats as stats


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

def flatten_and_clean(data):
    flat_data = data.flatten()
    clean_data = flat_data[~np.isnan(flat_data)]
    return clean_data

def combined_mask(uav_n_sentinel_data):
    masks = []
    for data in uav_n_sentinel_data:
        mask1 = ~np.isnan(data)
        mask2 = np.where(data < 1, True, False)
        mask3 = np.where(data > 0, True, False)

        mask = mask1 & mask2 & mask3

        masks.append(mask)

    mega_mask = masks[0]

    for mask in masks:
        mega_mask = mega_mask & mask

    return mega_mask

def apply_mask(mask,data):
    masked_common = np.where(mask, data, np.nan)
    return masked_common


# Red, NIR, SWIR2
sentinel_bands = [0,3,5]

sentinel_band_names = ['Red', 'NIR', 'SWIR2']

# Red, Green, Blue, NIR, Red-Edge
uav_bands = [0,1,2,3,4]


uav_band_names = ['Red',
             'Green',
             'Blue',
             'NIR',
             'Red-Edge']

for i,uav_date in enumerate(uav_dates):
    sentinel_date = sentinel_dates[i]

    uav_path = f"UAV Data/{uav_date} Aligned 10m Cropped.tif"
    sentinel_path = f"Sentinel Data/{sentinel_date} SENTINEL Cropped.tif"

    with rasterio.open(sentinel_path) as sentinel:
        for sentinel_band_index,sentinel_band_type in zip(sentinel_bands,sentinel_band_names):
            sentinel_band = sentinel.read(sentinel_band_index+1)
            sentinel_mask = ~np.isnan(sentinel_band)


            spearman_list = []
            data_dict = {}
            with rasterio.open(uav_path) as uav:
                for uav_band_index,uav_band_type in zip(uav_bands, uav_band_names):
                    uav_band = uav.read(uav_band_index+1)
                    uav_mask = ~np.isnan(uav_band)
                    mask = sentinel_mask & uav_mask

                    sentinel_masked = np.where(mask, sentinel_band, np.nan)
                    sentinel_flat = flatten_and_clean(sentinel_masked)

                    uav_masked = np.where(mask, uav_band, np.nan)
                    uav_flat = flatten_and_clean(uav_masked)

                    spearman_corr, _ = stats.spearmanr(sentinel_flat, uav_flat)
                    spearman_corr = abs(spearman_corr)
                    spearman_list.append(spearman_corr)

                    print(f"{sentinel_date} SENTINEL {sentinel_band_type} with {uav_date} UAV {uav_band_type}: {spearman_corr:.3}")

            data_list = sorted(zip(spearman_list, uav_band_names, uav_bands))
            data_list.reverse()

            _, sorted_bands, sorted_indices = list(zip(*data_list))
            print(sorted_bands)
            print(sorted_indices)

            np.save(f"Ranked Indices/{sentinel_date} {sentinel_band_type} Sorted Indices.npy", sorted_indices)


        

