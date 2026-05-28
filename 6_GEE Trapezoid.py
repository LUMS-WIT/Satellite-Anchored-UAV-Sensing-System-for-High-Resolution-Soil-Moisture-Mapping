import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

def flatten_and_clean(data):
    flat_data = data.flatten()
    clean_data = flat_data[~np.isnan(flat_data)]
    return clean_data

def combined_mask(uav_n_sentinel_data):
    masks = []
    for data in uav_n_sentinel_data:
        mask = ~np.isnan(data)
        masks.append(mask)

    mega_mask = masks[0]

    for mask in masks:
        mega_mask = mega_mask & mask

    return mega_mask

def apply_mask(mask,data):
    masked_common = np.where(mask, data, np.nan)
    return masked_common


folder_path = "GEE 2018-2023"

for file in os.listdir(folder_path):
    full_path = os.path.join(folder_path, file)

str_extend = []
ndvi_extend = []


for file in os.listdir(folder_path):
    sentinel_file = os.path.join(folder_path, file)

    with rasterio.open(sentinel_file) as sentinel:
        sentinel_red = sentinel.read(1)
        sentinel_green = sentinel.read(2)
        sentinel_blue = sentinel.read(3)
        sentinel_nir = sentinel.read(4)
        sentinel_swir1 = sentinel.read(5)
        sentinel_swir2 = sentinel.read(6)

        all_data = [sentinel_red,
                    sentinel_green,
                    sentinel_blue,
                    sentinel_nir,
                    sentinel_swir1,
                    sentinel_swir2]
        
        mask = combined_mask(all_data)

        sentinel_red = apply_mask(mask,sentinel_red)
        sentinel_nir = apply_mask(mask,sentinel_nir)
        sentinel_swir2 = apply_mask(mask,sentinel_swir2)

        sentinel_ndvi = (sentinel_nir - sentinel_red)/(sentinel_nir + sentinel_red)
        sentinel_str1 = (1-sentinel_swir1)**2/(2*sentinel_swir1)
        sentinel_str2 = (1-sentinel_swir2)**2/(2*sentinel_swir2)

        s_ndvi_flat = flatten_and_clean(sentinel_ndvi)
        s_str2_flat = flatten_and_clean(sentinel_str2)

        ndvi_mask = np.where((s_ndvi_flat>=0) & (s_ndvi_flat<=1), s_ndvi_flat, np.nan)
        str2_mask = np.where((s_str2_flat>=0) & (s_str2_flat<=10), s_str2_flat, np.nan)

        mask = ~np.isnan(ndvi_mask) & ~np.isnan(str2_mask)

        s_ndvi_flat = np.where(mask, s_ndvi_flat, np.nan)
        s_str2_flat = np.where(mask, s_str2_flat, np.nan)

        s_ndvi_flat = flatten_and_clean(s_ndvi_flat)
        s_str2_flat = flatten_and_clean(s_str2_flat)

        str_extend.extend(s_str2_flat)
        ndvi_extend.extend(s_ndvi_flat)





def ndvi_binning(ndvi_list, str_list, n_bins=100, min_points=5):

    ndvi = np.array(ndvi_list)
    STR = np.array(str_list)

    bins = np.linspace(ndvi.min(), ndvi.max(), n_bins + 1)

    dry_points = []
    wet_points = []

    for i in range(n_bins):

        mask = (ndvi >= bins[i]) & (ndvi < bins[i+1])

        if np.sum(mask) >= min_points:

            ndvi_vals = ndvi[mask]
            str_vals = STR[mask]

            mean_ndvi = np.mean(ndvi_vals)

            dry_str = np.min(str_vals)
            wet_str = np.max(str_vals)

            dry_points.append([mean_ndvi, dry_str])
            wet_points.append([mean_ndvi, wet_str])

    dry_points = np.array(dry_points)
    wet_points = np.array(wet_points)

    return dry_points, wet_points


def density_edges(ndvi, STR, bins=100, density_threshold=5):

    H, xedges, yedges = np.histogram2d(ndvi, STR, bins=bins)

    dry_points = []
    wet_points = []

    for i in range(len(xedges)-1):

        column = H[i]


        valid = np.where(column > density_threshold)[0]


        if len(valid) > 0:

            dry_bin = valid[0]
            wet_bin = valid[-1]

            ndvi_val = (xedges[i] + xedges[i+1]) / 2

            dry_str = (yedges[dry_bin] + yedges[dry_bin+1]) / 2
            wet_str = (yedges[wet_bin] + yedges[wet_bin+1]) / 2

            dry_points.append([ndvi_val, dry_str])
            wet_points.append([ndvi_val, wet_str])

    return np.array(dry_points), np.array(wet_points)


#dry, wet = ndvi_binning(ndvi_extend, str_extend)
#bins = np.sqrt(len(ndvi_extend))

#print(np.sqrt(len(ndvi_extend)))
#plt.hist2d(ndvi_extend, str_extend, bins=int(np.sqrt(len(ndvi_extend))))
plt.hist2d(ndvi_extend, str_extend, bins=100, cmap="gist_rainbow", norm=LogNorm())
plt.colorbar(label="Point Density")

plt.xlabel("NDVI")
plt.ylabel("STR")
plt.title("NDVI vs STR Density")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_Trapezoid.png")

plt.show()

#dry, wet = density_edges(ndvi_extend, str_extend, bins = int(np.sqrt(len(ndvi_extend))))
dry, wet = density_edges(ndvi_extend, str_extend, bins = 100)

dry_ndvi = dry[:,0]
dry_str = dry[:,1]

wet_ndvi = wet[:,0]
wet_str = wet[:,1]

dry_fit = np.polyfit(dry_ndvi, dry_str, 1)
wet_fit = np.polyfit(wet_ndvi, wet_str, 1)

print(dry_fit)
print(wet_fit)

dry_line = np.poly1d(dry_fit)
wet_line = np.poly1d(wet_fit)

dry_line2 = np.poly1d([1,0])
wet_line2 = np.poly1d([12,2])

sample = np.linspace(0, 1, 100)

plt.hist2d(ndvi_extend, str_extend, bins=100, cmap="gist_rainbow", norm=LogNorm())
plt.colorbar(label="Point Density")
plt.plot(sample, dry_line(sample), color="red", label="Dry Edge")
plt.plot(sample, wet_line(sample), color="blue", label="Wet Edge")
plt.xlabel("NDVI")
plt.ylabel("STR")
plt.title("NDVI vs STR Density")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_params1.png")
plt.show()


plt.hist2d(ndvi_extend, str_extend, bins=100, cmap="gist_rainbow", norm=LogNorm())
plt.colorbar(label="Point Density")
plt.plot(sample, dry_line2(sample), color="red", label="Dry Edge")
plt.plot(sample, wet_line2(sample), color="blue", label="Wet Edge")
plt.xlabel("NDVI")
plt.ylabel("STR")
plt.title("NDVI vs STR Density")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_params2.png")
plt.show()

#plt.scatter(ndvi_extend,str_extend,marker=',')
#plt.plot(x, dry_line(x), color="red", label="Dry Edge")
#plt.plot(x, wet_line(x), color="blue", label="Wet Edge")
#plt.legend()
#plt.xlabel("Sentinel NDVI")
#plt.ylabel("Sentinel STR")
#plt.title(f"2018 - 2023 Sentinel STR vs NDVI")
#plt.savefig(f"trapezoids/2018 - 2023 SENTINEL_STR2_vs_NDVI_AUTOMATED.png")
#plt.close()



'''
x = np.array(ndvi_extend)
y = np.array(str_extend)



#idx = np.random.choice(np.arange(len(x)), 80000, replace=False)
#x_sample = x[idx]
#y_sample = y[idx]

n_kde = min(10_000, len(x))
idx = np.random.choice(len(x), size=n_kde, replace=False)
xy_sub = np.vstack([x[idx], y[idx]])
kde = gaussian_kde(xy_sub)

# Evaluate density on all points using the subsampled KDE
xy_full = np.vstack([x, y])
z = kde(xy_full)

x_sample = x
y_sample = y


#xy = np.vstack([x_sample,y_sample])
#z = gaussian_kde(xy)(xy)



print("Plotting...")
fig, ax = plt.subplots()
sample = np.linspace(0, 1, 100)

#sc = ax.scatter(x_sample, y_sample, c=z, s=1, cmap="gist_rainbow_r", norm=LogNorm())
sc = ax.scatter(x_sample, y_sample, c=z, s=1, cmap="gist_rainbow_r")
ax.set_xlabel("Sentinel NDVI")
ax.set_ylabel("Sentinel STR")
ax.set_title("2018 - 2023 Sentinel STR vs NDVI")
ax.legend()
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Density / Z value")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_Trapezoid.png")
ax.cla()
plt.cla()

fig, ax = plt.subplots()
#sc = ax.scatter(x_sample, y_sample, c=z, s=1, cmap="gist_rainbow_r", norm=LogNorm())
sc = ax.scatter(x_sample, y_sample, c=z, s=1, cmap="gist_rainbow_r")
ax.set_ylim([-1,15])
ax.plot(sample, dry_line(sample), color="red", label="Dry Edge")
ax.plot(sample, wet_line(sample), color="blue", label="Wet Edge")
ax.set_xlabel("Sentinel NDVI")
ax.set_ylabel("Sentinel STR")
ax.set_title("2018 - 2023 Sentinel STR vs NDVI")
ax.legend()
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Density / Z value")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_params1.png")
ax.cla()
plt.cla()

fig, ax = plt.subplots()
sc = ax.scatter(x_sample, y_sample, c=z, s=1, cmap="gist_rainbow_r", norm=LogNorm())
ax.set_ylim([-1,15])
ax.plot(sample, dry_line2(sample), color="red", label="Dry Edge")
ax.plot(sample, wet_line2(sample), color="blue", label="Wet Edge")
ax.set_xlabel("Sentinel NDVI")
ax.set_ylabel("Sentinel STR")
ax.set_title("2018 - 2023 Sentinel STR vs NDVI")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Density / Z value")
plt.savefig("trapezoids/2018 - 2023 SENTINEL_STR_vs_NDVI_Heatmap_params2.png")
ax.cla()
plt.cla()

'''


        


        

        

         
        














        


