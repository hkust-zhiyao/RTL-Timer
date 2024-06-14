import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json, pickle
from scipy.stats import stats
import sys
from pathlib import Path
folder = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(folder))
from utils.statistics import *


def load_data():
    dc_label_data = f'/data/wenjifang/vlg2netlist/dc_stat/scr2/feat_label_data/all/label_all_SYN_delay.pkl'
    feat_data = f'/data/wenjifang/vlg2netlist/dc_stat/scr2/feat_label_data/all/feat_all_SYN_delay.pkl'
    std = StandardScaler()
    with open(dc_label_data, 'rb') as f:
        dc_data_dict = pickle.load(f)
    df1 = pd.DataFrame(dc_data_dict)
    # df1 = df1.T
    
    with open(feat_data, 'rb') as f:
        feat_dict = pickle.load(f)
    df2 = pd.DataFrame(feat_dict)
    
    print(df2)
    print(df1.shape)
    print(df2.shape)
    # df2.drop(df2.columns[[0,1,2,8,10]], axis=1, inplace=True) ### for comb cell
    # df2.drop(df2.columns[[1,2,3,4,5,6,7,8,9,10]], axis=1, inplace=True) ### for seq cell

    return df1, df2



def draw_fig_kf(title, y_test, y_pred, method, train_test):
    mse = metrics.mean_squared_error(normalization(y_test), normalization(y_pred), squared=True)
    mse = round(mse, 3)
    mspe = mspe_cal(y_test, y_pred)
    mspe = round(mspe, 3)
    rmse = metrics.mean_squared_error(normalization(y_test), normalization(y_pred), squared=False)
    rmse = round(rmse, 3)
    rmspe = rmspe_cal(y_test, y_pred)
    rmspe = round(rmspe, 3)
    mape_val = mape(y_pred, y_test)
    mape_val = round(mape_val)
    print("MAPE:", mape_val)
    print("RMSE:", rmse)
    r, p = stats.pearsonr(y_test, y_pred)
    r = round(r, 3)
    print("R:", r)
    r2 = metrics.r2_score(y_test, y_pred)
    r2 = round(r2, 3)
    print("R2:", r2)
    fig, ax = plt.subplots(tight_layout = True)
    if train_test == 'Train':
        ax.scatter(y_test, y_pred, c="b", alpha= 0.003)
    elif train_test == 'Test':
        ax.scatter(y_test, y_pred, c="orange", alpha= 0.003)
    else:
        assert False
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], ls="--", c='r', alpha = 0.2)
    ax.set_xlabel('Measured', fontproperties='Times New Roman', size=14)
    ax.set_ylabel('Predicted', fontproperties='Times New Roman', size=14)
    bbox = dict(boxstyle="round", fc='1', alpha=0.5)
    plt.text( 0.05, 0.75,  fontdict={'family':'Times New Roman'}, s=f'R = {r}\nMAPE = {mape_val}%\nRMSE = {rmse}', 
                    transform=ax.transAxes, size=14, bbox=bbox)
    if title != 'WNS':
        plt.xscale('log')
        plt.yscale('log')
    plt.title(f'{title} {train_test}', fontdict={'family':'Times New Roman', 'size':24})
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.show()
    plt.savefig(f"../fig/{method}_{title}.png", dpi=300)
