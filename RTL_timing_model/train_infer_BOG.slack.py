# from preprocess import *
import xgboost as xgb
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
## import linear_model
from sklearn.linear_model import LinearRegression

import os, time, json, copy, pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from random import shuffle
from stat_ import *

def testing(design_name, xgbr):
    feat_label_dir = "../preprocess/feat_label_timing"


    feat_vec, label_vec = [], []
    with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.pkl", "rb") as f:
        feat_label_design = pickle.load(f)
    for reg_dct in feat_label_design:
        feat = []
        # feat.extend(reg_dct[f'feat_design'])
        feat.extend(reg_dct[f'feat_path'])
        feat_vec.append(feat)
        label_vec.append(reg_dct[f"label_slack"])
    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)
    
    df_feat = pd.DataFrame(feat_arr)
    df_label = pd.DataFrame(label_arr)
    # print(df_feat)
    # print(df_label)
    # input()
    pred = xgbr.predict(df_feat).flatten()



    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred, label_arr)
    if not os.path.exists(f"./fig/{cmd}{label_cmd}_design"):
        os.makedirs(f"./fig/{cmd}{label_cmd}_design")
    draw_scatter_plot(pred, label_arr, f"./fig/{cmd}{label_cmd}_design/{design_name}.png", title=f"{cmd}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

    
    if np.isnan(r_val):
        r_val = 1

    total_slack_pred = np.sum(pred)
    total_slack_real = np.sum(label_arr)

    pred_lst.append(total_slack_pred)
    real_lst.append(total_slack_real)
    pred_module_lst.extend(pred.tolist())
    real_module_lst.extend(label_arr.tolist())

    return r_val, mape_val, rrse_val, mae_val, total_slack_pred, total_slack_real


# def data_augment(feat_arr, label_arr):
#     ### sample k data from the training data and add the features and labels to form a new data point
#     k = 2
#     for i in range(10):
#         idx = np.random.randint(0, len(feat_arr), k)
#         feat = feat_arr[idx]
#         label = label_arr[idx]
#         # print(feat, label)
#         feat = np.mean(feat, axis=0)
#         label = np.mean(label, axis=0)
    
#         feat_arr = np.append(feat_arr, [feat], axis=0)
#         label_arr = np.append(label_arr, [label], axis=0)

#     return feat_arr, label_arr

def training(train_lst):
    feat_label_dir = "../preprocess/feat_label_timing"
    feat_vec, label_vec = [], []
    for design_name in train_lst:
        feat_vec, label_vec = [], []
        with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.pkl", "rb") as f:
            feat_label_design = pickle.load(f)
        for reg_dct in feat_label_design:
            feat = []
            # feat.extend(reg_dct[f'feat_design'])
            feat.extend(reg_dct[f'feat_path'])
            # print(len(feat))
            
            feat_vec.append(feat)
            label_vec.append(reg_dct[f"label_slack"])

    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)

    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)
    
    df_feat = pd.DataFrame(feat_arr)
    df_label = pd.DataFrame(label_arr)
    print(df_feat.shape, df_label.shape)

    xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=50, nthread=25)
    xgbr.fit(df_feat, df_label)

    return xgbr


def train_test(train_lst, test_lst):

    r_lst, mape_lst, rrse_lst, mae_lst = [], [], [], []
    total_pwr_pred_lst, total_pwr_real_lst = [], []



    
    
    xgbr = training(train_lst)
    for design in test_lst:
        print(f"Design {design} ...")
        print(f"Testing ...")
        r_val, mape_val, rrse_val, mae_val, total_pwr_pred, total_pwr_real = testing(design, xgbr)
        print(f"R: {r_val}")
        print(f"MAPE: {mape_val}")
        print(f"RRSE: {rrse_val}")
        print(f"Total Slack Prediction: {total_pwr_pred}")
        print(f"Total Slack Real: {total_pwr_real}")
        r_lst.append(r_val)
        mape_lst.append(mape_val)
        rrse_lst.append(rrse_val)
        mae_lst.append(mae_val)
        total_pwr_pred_lst.append(total_pwr_pred)
        total_pwr_real_lst.append(total_pwr_real)
        print('\n')

    print(f"BOG ({cmd}) Design Average")
    print(f"Average R: {round(np.mean(r_lst), 4)}")
    print(f"Average MAPE: {round(np.mean(mape_lst), 4)}")
    print(f"Average RRSE: {round(np.mean(rrse_lst), 4)}")
    print(f"Average MAE: {round(np.mean(mae_lst), 4)}")

    print(f"\nTotal Power Average")
    r_val, mape_val, rrse_val, mape_val = regression_metrics(total_pwr_pred_lst, total_pwr_real_lst)
    
    ## save powr list
    


if __name__ == '__main__':



    global cmd, label_cmd
    cmd = 'sog'
    # cmd = "xag"
    # cmd = "aig"
    # cmd = "aimg"
    label_cmd = "_init"
    # label_cmd = "_route"
    # label_cmd = "_route_calibre"
    # label_cmd = ""

    with open ("./design_js/train_lst.json", "r") as f:
        train_lst = json.load(f)
    with open ("./design_js/test_lst.json", "r") as f:
        test_lst = json.load(f)

    ## k-fold cross validation
    # kf = KFold(n_splits=5,


    # training(train_lst, test_lst)

    # for cmd in ["sog", "xag", "aig", "aimg"]:
        global pred_lst, real_lst, pred_module_lst, real_module_lst
        pred_lst, real_lst, pred_module_lst, real_module_lst = [], [], [], []
        train_test(train_lst, test_lst)
        if not os.path.exists(f"./fig/{cmd}{label_cmd}"):
            os.makedirs(f"./fig/{cmd}{label_cmd}")
        r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_lst, real_lst)
        draw_scatter_plot(pred_lst, real_lst, f"./fig/{cmd}{label_cmd}/{cmd}{label_cmd}.png", title=f"{cmd}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

        r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_module_lst, real_module_lst)
        draw_scatter_plot(pred_module_lst, real_module_lst, f"./fig/{cmd}{label_cmd}/{cmd}_module{label_cmd}.png", title=f"{cmd}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")


        # with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_pred.pkl", "wb") as f:
        #     pickle.dump(pred_lst, f)
        # with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_real.pkl", "wb") as f:
        #     pickle.dump(real_lst, f)

        # with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_pred_module.pkl", "wb") as f:
        #     pickle.dump(pred_module_lst, f)
        # with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_real_module.pkl", "wb") as f:
        #     pickle.dump(real_module_lst, f)