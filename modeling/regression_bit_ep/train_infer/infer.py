# from preprocess import *
import xgboost as xgb
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor


import os, time, json, copy, pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from random import shuffle
from stat_ import *
design_json = "/data/wenjifang/AIG_analyzer/LS-benchmark/design_timing_rgb.json"



def testing(test_lst):
    feat_label_dir = "/home/coguest5/RTL-Timer/modeling/feat_label/bit-wise"
    graph_feat_dir = "/home/coguest5/RTL-Timer/modeling/feat_label/graph_feat"
    for design_name in test_lst:
        with open (f"{feat_label_dir}/{design_name}.pkl", "rb") as f:
            feat_label_pair = pickle.load(f)
        
        feat_dict, label_dict = feat_label_pair

        # print(feat_dict)
        # print(label_dict)

        feat_arr = np.array(list(feat_dict.values()))
        print(feat_arr.shape)
        label_arr = np.array(list(label_dict.values()))
        label_arr = label_arr[:,0]
        
    
    
        df_feat = pd.DataFrame(feat_arr)
        df_feat.drop(df_feat.columns[[25]], axis=1, inplace=True)

        save_path = f"../saved_model/bit_ep_model_{design_name}.pkl"
        print('Testing ...')
        with open (save_path, "rb") as f:
            xgbr = pickle.load(f)
        pred = xgbr.predict(df_feat)

        regression_metrics(pred, label_arr)

        

def get_design_lst(bench):
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        design_lst.append(k)



if __name__ == '__main__':
    bench_list_all = ['iscas','itc','opencores','VexRiscv','chipyard', 'riscvcores','NVDLA']

    design_name = 'TinyRocket'
    # design_name = ""

    global design_lst, phase
    design_lst = []
    phase = 'SYN'

    with open ("./design_js/test_lst.json", "r") as f:
        test_lst = json.load(f)
    
    testing(test_lst)

# def run_xgb_label(x_train, y_train):

#     y_train = y_train.astype(float)
#     xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=100, nthread=25)

#     xgbr.fit(x_train, y_train)

#     # y_pred = xgbr.predict(x_test)
#     # y_pred2 = xgbr.predict(x_train)
#     with open ("../saved_model/ep_model_sog.pkl", "wb") as f:
#         pickle.dump(xgbr, f)




# if __name__ == '__main__':
#     label_data, feat_data = load_data()
#     x = feat_data
#     y = label_data ## seq label

#     # kFold_train(x, y, 'EP Model')
#     run_xgb_label(x, y)

