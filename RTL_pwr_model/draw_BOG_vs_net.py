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

def run_one_design(design_name):
    feat_label_dir = "../preprocess/feat_label_pwr"


    total_pwr_pred, total_pwr_real = 0, 0

    feat_vec, label_vec = [], []
    with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.json", "r") as f:
        feat_label_design = json.load(f)
    for module_dct in feat_label_design:
        # feat_vec.append(module_dct[f'bog_{pwr_comp}'])
        # label_vec.append(module_dct[f"label_{pwr_comp}"])

        feat_vec.append(module_dct[f'bog_pwr'])
        label_vec.append(module_dct[f"label"])
    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)

    r_val, mape_val, rrse_val, mae_val  = regression_metrics(feat_arr, label_arr)
    
    # draw_scatter_plot(feat_arr, label_arr, f"./fig_BOG_vs_net/{cmd}_{pwr_comp}{label_cmd}_{design_name}.png", title=f"./fig_BOG_vs_net/{cmd}_{pwr_comp}{label_cmd}/{design_name}.png, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}")

    if np.isnan(r_val):
        r_val = 1

    total_pwr_pred = np.sum(feat_arr)
    total_pwr_real = np.sum(label_arr)

    pred_lst.append(total_pwr_pred)
    real_lst.append(total_pwr_real)
    pred_module_lst.extend(feat_arr.tolist())
    real_module_lst.extend(label_arr.tolist())

    return r_val, mape_val, rrse_val, mae_val, total_pwr_pred, total_pwr_real



    


if __name__ == '__main__':

    global pwr_comp, pwr_tpe
    # pwr_tpe = "comb"
    pwr_tpe = "reg"
    # pwr_comp = f"{pwr_tpe}_dyn"
    pwr_comp = f"clk_dyn"

    pwr_comp = "total"


    global cmd, label_cmd
    # cmd = 'sog'
    # cmd = "xag"
    cmd = "aig"
    # cmd = "aimg"
    label_cmd = "_init"
    # label_cmd = "_route"
    # label_cmd = "_route_calibre"
    # label_cmd = ""

    with open ("./design_js/design_lst.json", "r") as f:
        design_lst = json.load(f)

    ## k-fold cross validation
    # kf = KFold(n_splits=5,


    # training(train_lst, test_lst)

    # for cmd in ["sog", "xag", "aig", "aimg"]:
    global pred_lst, real_lst, pred_module_lst, real_module_lst
    pred_lst, real_lst, pred_module_lst, real_module_lst = [], [], [], []
    
    for design in design_lst:
        run_one_design(design)

    # draw_scatter_plot(pred_lst, real_lst, f"./saved_fig_BOG_vs_net/{cmd}_{pwr_comp}{label_cmd}.png", title=f"{cmd} {pwr_comp}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_module_lst, real_module_lst)
    draw_scatter_plot(pred_module_lst, real_module_lst, f"./fig/BOG_vs_net/{cmd}_{pwr_comp}_module{label_cmd}.png", title=f"{cmd} {pwr_comp}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_lst, real_lst)
    draw_scatter_plot(pred_lst, real_lst, f"./fig/BOG_vs_net/{cmd}_{pwr_comp}{label_cmd}.png", title=f"{cmd} {pwr_comp}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

    # with open (f"./fig_BOG_vs_net/pwr_{cmd}_{pwr_comp}{label_cmd}_pred.pkl", "wb") as f:
    #     pickle.dump(pred_lst, f)
    # with open (f"./fig_BOG_vs_net/pwr_{cmd}_{pwr_comp}{label_cmd}_real.pkl", "wb") as f:
    #     pickle.dump(real_lst, f)

    # with open (f"./fig_BOG_vs_net/pwr_{cmd}_{pwr_comp}{label_cmd}_pred_module.pkl", "wb") as f:
    #     pickle.dump(pred_module_lst, f)
    # with open (f"./fig_BOG_vs_net/pwr_{cmd}_{pwr_comp}{label_cmd}_real_module.pkl", "wb") as f:
    #     pickle.dump(real_module_lst, f)