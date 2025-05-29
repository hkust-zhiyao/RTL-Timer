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
    feat_label_dir = "../preprocess/feat_label_pwr"


    total_pwr_pred, total_pwr_real = 0, 0

    feat_vec, label_vec = [], []
    with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.json", "r") as f:
        feat_label_design = json.load(f)
    for module_dct in feat_label_design:
        feat = module_dct[f'feat_{pwr_tpe}']
        feat.append(module_dct[f'bog_{pwr_comp}'])
        feat_vec.append(feat)
        label_vec.append(module_dct[f"label_{pwr_comp}"])
    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)
    
    df_feat = pd.DataFrame(feat_arr)
    # df_feat.drop(df_feat.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30,31,32]], axis=1, inplace=True)
    print(df_feat.shape)
    # print(df_feat)
    # input()

    pred = xgbr.predict(df_feat).flatten()
    ## pred [[a, b,]]

    # print(pred.shape)
    with open (f"./pred/{design_name}_{cmd}.json", "w") as f:
        json.dump(pred.tolist(), f, indent=4)
    # input()



    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred, label_arr)
    # if not os.path.exists(f"./fig/{cmd}_{pwr_comp}{label_cmd}"):
    #     os.makedirs(f"./fig/{cmd}_{pwr_comp}{label_cmd}")
    # draw_scatter_plot(pred, label_arr, f"./fig/{cmd}_{pwr_comp}{label_cmd}/{design_name}.png", title=f"./fig/{cmd}_{pwr_comp}{label_cmd}/{design_name}.png, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}")

    

    


    ## r_val is 1 if r_val is nan
    if np.isnan(r_val):
        r_val = 1

    total_pwr_pred = np.sum(pred)
    total_pwr_real = np.sum(label_arr)

    pred_lst.append(total_pwr_pred)
    real_lst.append(total_pwr_real)
    pred_module_lst.extend(pred.tolist())
    real_module_lst.extend(label_arr.tolist())

    return r_val, mape_val, rrse_val, mae_val, total_pwr_pred, total_pwr_real


def data_augment(feat_arr, label_arr):
    ### sample k data from the training data and add the features and labels to form a new data point
    k = 2
    for i in range(10):
        idx = np.random.randint(0, len(feat_arr), k)
        feat = feat_arr[idx]
        label = label_arr[idx]
        # print(feat, label)
        feat = np.mean(feat, axis=0)
        label = np.mean(label, axis=0)
    
        feat_arr = np.append(feat_arr, [feat], axis=0)
        label_arr = np.append(label_arr, [label], axis=0)

    return feat_arr, label_arr

def training(train_lst):
    feat_label_dir = "../preprocess/feat_label_pwr"
    feat_vec, label_vec = [], []
    for design_name in train_lst:
        with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.json", "r") as f:
            feat_label_design = json.load(f)
        for module_dct in feat_label_design:
            feat = module_dct[f'feat_{pwr_tpe}']
            feat.append(module_dct[f'bog_{pwr_comp}'])
            feat_vec.append(feat)
            label_vec.append(module_dct[f"label_{pwr_comp}"])


    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)

    feat_arr, label_arr = data_augment(feat_arr, label_arr)

    # feat_all.extend(list(feat_arr))
    # label_all.extend(list(label_arr))
    
    
    df_feat = pd.DataFrame(feat_arr)

    

    # df_feat.drop(df_feat.columns[[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,26,27,28,29,30,31,32]], axis=1, inplace=True)
    df_label = pd.DataFrame(label_arr)
    print(df_feat.shape, df_label.shape)

    xgbr = xgb.XGBRegressor(n_estimators=2000, max_depth=500, nthread=25)
    # xgbr = LinearRegression()
    xgbr.fit(df_feat, df_label)

    return xgbr

    # save_dir = "./saved_model/"
    # os.system(f"rm -rf {save_dir}/*")
    # print('Training ...')
    # with open (f"{save_dir}/pwr_model.pkl", "wb") as f:
    #     pickle.dump(xgbr, f)
    # for test_name in test_lst:
    #     os.system(f"cp -r {save_dir}/ep_model.pkl {save_dir}/bit_ep_model_{test_name}.pkl")
    # os.system(f"rm -rf {save_dir}/ep_model.pkl")
    # print('Finish!')


def k_fold(design_lst):

    r_lst, mape_lst, rrse_lst, mae_lst = [], [], [], []
    total_pwr_pred_lst, total_pwr_real_lst = [], []
    total_pwr_module_pred_lst, total_pwr_module_real_lst = [], []

    print(len(design_lst))


    

    for design in design_lst:
        print(f"Design {design} ...")
        ## rest_lst = copy.deepcopy(design_lst)
        rest_lst = design_lst.copy()
        rest_lst.remove(design)
        print(f"Training ...")
        xgbr = training(rest_lst)
        print(f"Testing ...")
        r_val, mape_val, rrse_val, mae_val, total_pwr_pred, total_pwr_real = testing(design, xgbr)
        print(f"R: {r_val}")
        print(f"MAPE: {mape_val}")
        print(f"RRSE: {rrse_val}")
        print(f"Total Power Prediction: {total_pwr_pred}")
        print(f"Total Power Real: {total_pwr_real}")
        r_lst.append(r_val)
        mape_lst.append(mape_val)
        rrse_lst.append(rrse_val)
        mae_lst.append(mae_val)
        total_pwr_pred_lst.append(total_pwr_pred)
        total_pwr_real_lst.append(total_pwr_real)
        print('\n')

    print(f"BOG ({cmd}) {pwr_tpe} Design Average")
    print(f"Average R: {round(np.mean(r_lst), 4)}")
    print(f"Average MAPE: {round(np.mean(mape_lst), 4)}")
    print(f"Average RRSE: {round(np.mean(rrse_lst), 4)}")
    print(f"Average MAE: {round(np.mean(mae_lst), 4)}")

    print(f"\nTotal Power Average")
    r_val, mape_val, rrse_val, mape_val = regression_metrics(total_pwr_pred_lst, total_pwr_real_lst)

    ## save powr list
    


if __name__ == '__main__':

    global pwr_comp, pwr_tpe
    global cmd, label_cmd

    ## ========= change the power type here =========
    ## ------ 1. power type ------
    ## combinational power
    # pwr_tpe = "comb"
    # pwr_comp = f"{pwr_tpe}_dyn"

    ## register power
    # pwr_tpe = "reg"
    # pwr_comp = f"{pwr_tpe}_dyn"

    ## clock network power
    pwr_tpe = "reg"
    pwr_comp = f"clk_dyn"
    
    ## ------ 2. BOG type (default: SOG) ------
    cmd = 'sog'
    # cmd = "xag"
    # cmd = "aig"
    # cmd = "aimg"

    ## ------ 3. label stage (init: post-syn, route: post-layout) ------
    label_cmd = "_init"
    # label_cmd = "_route"

    ## =================== End ==============================


    with open ("./design_js/design_lst.json", "r") as f:
        design_lst = json.load(f)

    global pred_lst, real_lst, pred_module_lst, real_module_lst
    pred_lst, real_lst, pred_module_lst, real_module_lst = [], [], [], []
    k_fold(design_lst)
    if not os.path.exists(f"./fig/{cmd}{label_cmd}"):
        os.makedirs(f"./fig/{cmd}{label_cmd}")
    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_lst, real_lst)
    draw_scatter_plot(pred_lst, real_lst, f"./fig/{cmd}{label_cmd}/{cmd}_{pwr_comp}{label_cmd}.png", title=f"{cmd} {pwr_comp}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")

    r_val, mape_val, rrse_val, mae_val  = regression_metrics(pred_module_lst, real_module_lst)
    draw_scatter_plot(pred_module_lst, real_module_lst, f"./fig/{cmd}{label_cmd}/{cmd}_{pwr_comp}_module{label_cmd}.png", title=f"{cmd} {pwr_comp}, R={round(np.mean(r_val), 2)}, MAPE={round(np.mean(mape_val), 1)}, RRSE={round(np.mean(rrse_val), 3)}, MAE={round(np.mean(mae_val), 3)}")


    with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_pred.pkl", "wb") as f:
        pickle.dump(pred_lst, f)
    with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_real.pkl", "wb") as f:
        pickle.dump(real_lst, f)

    with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_pred_module.pkl", "wb") as f:
        pickle.dump(pred_module_lst, f)
    with open (f"./saved_pwr/pwr_{cmd}_{pwr_comp}{label_cmd}_real_module.pkl", "wb") as f:
        pickle.dump(real_module_lst, f)