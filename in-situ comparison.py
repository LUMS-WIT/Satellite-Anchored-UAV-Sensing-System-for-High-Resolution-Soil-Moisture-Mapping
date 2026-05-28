import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
in_situ_data = [pd.read_csv(f"in-situ sensor data/plot{i}.csv") for i in range(5)]

auto = pd.read_excel(f"in-situ sensor data/auto.xlsx")
manual = pd.read_excel(f"in-situ sensor data/manual.xlsx")

#theta_d = 0.15
#theta_w = 0.36

def ubrmse(real,pred):
    real = np.array(real)
    pred = np.array(pred)
    real_mean = np.mean(real)
    pred_mean = np.mean(pred)
    bias = pred_mean - real_mean
    unb_pred = pred - bias
    ubrmse = np.sqrt(np.mean((real - unb_pred)**2))
    return ubrmse

def get_common_dates_df(df1, df2, date_col=None, suffixes=("_1", "_2")):  
    if date_col is None:
        common_idx = df1.index.intersection(df2.index)
        return df1.loc[common_idx].join(df2.loc[common_idx], how="inner", lsuffix=suffixes[0], rsuffix=suffixes[1])
    else:
        df1 = df1.copy()
        df2 = df2.copy()
        df1[date_col] = pd.to_datetime(df1[date_col])
        df2[date_col] = pd.to_datetime(df2[date_col])
        
        merged = df1.merge(df2, on=date_col, how="inner", suffixes=suffixes)
        return merged
    
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


def nan_remover(list1, list2):
    l1 = np.array(list1)
    l2 = np.array(list2)

    m1 = ~np.isnan(l1)
    m2 = ~np.isnan(l2)

    mask = m1 & m2


    return l1[mask],l2[mask]
    


for data in in_situ_data:
    dates = data['TimeStamp']
    dates_converted = []
    for d in dates:
        try:
            dt = datetime.strptime(d, "%d/%m/%Y %H:%M")
        except ValueError:
            dt = datetime.strptime(d, "%d/%m/%Y")
        dates_converted.append(str(dt.date()))
    data['Dates'] = dates_converted
    data['VWC'] = data['VolumetricWaterContent1']/100

in_situ_auto_all = []
auto_pred_all = []

in_situ_manual_all = []
manual_pred_all = []

auto_box = []
manual_box = []

for i in range(5):
    auto_df = auto[['Dates',f"plot_{i}"]]
    manual_df = manual[['Dates',f"plot_{i}"]]

    in_situ = in_situ_data[i]

    auto_merged_df = get_common_dates_df(auto_df, in_situ, 'Dates')
    manual_merged_df = get_common_dates_df(manual_df, in_situ, 'Dates')

    in_situ = auto_merged_df['VWC']

    auto_pred = auto_merged_df[f"plot_{i}"]
    manual_pred = manual_merged_df[f"plot_{i}"]

    auto_pred, in_situ_auto = nan_remover(auto_pred, in_situ)
    manual_pred, in_situ_manual = nan_remover(manual_pred, in_situ)

    auto_interval = interval(auto_pred, in_situ_auto)
    manual_interval = interval(manual_pred, in_situ_manual)


    in_situ_auto_all.extend(in_situ_auto)
    auto_pred_all.extend(auto_pred)

    in_situ_manual_all.extend(in_situ_manual)
    manual_pred_all.extend(manual_pred)

    auto_mae = mean_absolute_error(auto_pred, in_situ_auto)
    auto_r,_ = pearsonr(auto_pred, in_situ_auto)

    manual_mae = mean_absolute_error(manual_pred, in_situ_manual)
    manual_r,_ = pearsonr(manual_pred, in_situ_manual)

    ubrmse_auto = ubrmse(auto_pred, in_situ_auto)
    ubrmse_manual = ubrmse(manual_pred, in_situ_manual)

    print(f"Auto MAE: {round(auto_mae,4)} Auto R: {round(auto_r,4)} Auto ubrmse: {round(ubrmse_auto,4)} Auto Interval: {list(auto_interval)}")
    print(f"{round(auto_mae,4)} & {round(ubrmse_auto,4)} & {round(auto_r,4)} & {list(auto_interval)}")

    print(f"Manual MAE: {round(manual_mae,4)} Manual R: {round(manual_r,4)} Manual ubrmse: {round(ubrmse_manual,4)} Manual Interval: {list(manual_interval)}")
    print(f"{round(manual_mae,4)} & {round(ubrmse_manual,4)} & {round(manual_r,4)} & {list(manual_interval)}")

    auto_box.append([auto_mae, auto_interval[0], auto_interval[1]])
    manual_box.append([manual_mae, manual_interval[0], manual_interval[1]])
    print("-"*50)


    #plt.plot(in_situ_auto, auto_pred, '.')
    #plt.plot([theta_d, theta_w], [theta_d, theta_w], 'k--', label='1:1 Line')
    #plt.grid(False)
    #plt.show()

auto_mae_all = mean_absolute_error(auto_pred_all, in_situ_auto_all)
auto_r_all,_ = pearsonr(auto_pred_all, in_situ_auto_all)
auto_ubrmse_all = ubrmse(auto_pred_all, in_situ_auto_all)
auto_all_interval = interval(auto_pred_all, in_situ_auto_all)

print(f"Auto MAE All: {round(auto_mae_all,4)} Auto R All: {round(auto_r_all,4)} Auto uBRMSE All{round(auto_ubrmse_all,4)} Auto Interval: {list(auto_all_interval)}")


manual_mae_all = mean_absolute_error(manual_pred_all, in_situ_manual_all)
manual_r_all,_ = pearsonr(manual_pred_all, in_situ_manual_all)
manual_ubrmse_all = ubrmse(manual_pred_all, in_situ_manual_all)
manual_all_interval = interval(manual_pred_all, in_situ_manual_all)

print(f"Manual MAE All: {round(manual_mae_all,4)} Manual R All: {round(manual_r_all,4)} Manual uBRMSE All{round(manual_ubrmse_all,4)} Manual Interval: {list(manual_all_interval)}")
#auto_mae_all = mean_absolute_error(auto_pred_all, in_situ_auto_all)
#auto_r_all,_ = pearsonr(auto_pred_all, in_situ_auto_all)
#print(f"Auto MAE All: {round(auto_mae_all,4)} Auto R All: {round(auto_r_all,4)}")

labels = ["Plot 1","Plot2","Plot 3","Plot 4","Plot 5"]

x = [i for i in range(len(auto_box))]

print(x)
fig, ax = plt.subplots()
ax.set_xticks(x)                      # set tick positions
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0,0.12)

means, error_low, error_high = list(zip(*auto_box))
means = list(means)
error_bs = (np.array(error_high) - np.array(error_low))/2
ax.set_ylabel("Mean Absolute Error")
print(means)
print(error_bs)
ax.bar(x,means,width=0.4, yerr=error_bs, capsize=10,color='lightskyblue')
plt.savefig("trapezoids/Auto Bar plot.png")
plt.close()

fig, ax = plt.subplots()
ax.set_xticks(x)                      # set tick positions
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0,0.12)

means, error_low, error_high = list(zip(*manual_box))
means = list(means)
error_bs = (np.array(error_high) - np.array(error_low))/2
ax.set_ylabel("Mean Absolute Error")
print(means)
print(error_bs)
ax.bar(x,means,width=0.4, yerr=error_bs, capsize=10,color='lightskyblue')
plt.savefig("trapezoids/Manual Bar plot.png")
