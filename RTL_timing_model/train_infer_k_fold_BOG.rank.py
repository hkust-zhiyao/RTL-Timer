# from preprocess import *
import xgboost as xgb
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
## import linear_model
from sklearn.linear_model import LinearRegression

import os, time, json, copy, pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from multiprocessing import Pool
from random import shuffle
from stat_ import *

def testing(design_name, xgbr):
    feat_label_dir = "../preprocess/feat_label_timing"


    feat_vec, label_vec = [], []
    reg_lst = []
    with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.pkl", "rb") as f:
        feat_label_design = pickle.load(f)
    for idx, reg_dct in enumerate(feat_label_design):
        if idx <= len(feat_label_design) * 0.25:
            rank_label = 0
        elif len(feat_label_design) * 0.25 < idx <= len(feat_label_design) * 0.5:
            rank_label = 1
        elif len(feat_label_design) * 0.5 < idx <= len(feat_label_design) * 0.75:
            rank_label = 2
        else:
            rank_label = 3
        feat = []
        feat.extend(reg_dct[f'feat_design'])
        feat.extend(reg_dct[f'feat_path'])
        feat_vec.append(feat)
        label_vec.append(rank_label)
        reg_lst.append(reg_dct['name'])
    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)
    
    df_feat = pd.DataFrame(feat_arr)
    df_label = pd.DataFrame(label_arr)
    # print(df_feat)
    # print(df_label)
    # input()
    pred = xgbr.predict(df_feat).flatten()



    cover = coverage_rank_num(pred, label_arr)
    cover_all_lst.append(cover)



def training(train_lst):
    feat_label_dir = "../preprocess/feat_label_timing"
    feat_vec, label_vec = [], []
    group_all = []
    for design_name in train_lst:
        # feat_vec, label_vec = [], []
        with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.pkl", "rb") as f:
            feat_label_design = pickle.load(f)
        for idx, reg_dct in enumerate(feat_label_design):
            if idx <= len(feat_label_design) * 0.25:
                rank_label = 0
            elif len(feat_label_design) * 0.25 < idx <= len(feat_label_design) * 0.5:
                rank_label = 1
            elif len(feat_label_design) * 0.5 < idx <= len(feat_label_design) * 0.75:
                rank_label = 2
            else:
                rank_label = 3
            feat = []
            feat.extend(reg_dct[f'feat_design'])
            feat.extend(reg_dct[f'feat_path'])
            # print(len(feat))
            
            feat_vec.append(feat)
            label_vec.append(rank_label)
        group_all.append(len(feat_label_design))

    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)

    feat_arr = np.array(feat_vec)
    label_arr = np.array(label_vec)
    
    df_feat = pd.DataFrame(feat_arr)
    df_label = pd.DataFrame(label_arr)
    print(df_feat.shape, df_label.shape)

    # xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=50, nthread=25)

    ranker = xgb.XGBRanker(  
                            tree_method='hist',
                            device="cuda",
                            booster='gbtree',
                            objective='rank:pairwise',
                            random_state=42, 
                            learning_rate=0.1,
                            colsample_bytree=0.9, 
                            eta=0.05, 
                            max_depth=30, 
                            n_estimators=100, 
                            subsample=0.75 
                            )
    groups = np.array(group_all).reshape(-1, 1)
    print(groups.shape)
    grp_sum = np.sum(groups)
    print(f"Total groups: {grp_sum}")
    ranker.fit(df_feat, df_label, group=group_all)

    return ranker


def k_fold(design_lst):


    print(len(design_lst))

    for design in design_lst:
        print(f"Design {design} ...")
        ## rest_lst = copy.deepcopy(design_lst)
        rest_lst = design_lst.copy()
        rest_lst.remove(design)
        print(f"Training ...")
        xgbr = training(rest_lst)
        print(f"Testing ...")
        testing(design, xgbr)


    


if __name__ == '__main__':


    ## ========= change the power type here =========
    global cmd, label_cmd

    ## ------ 1. BOG type (default: SOG) ------
    cmd = 'sog'
    # cmd = "xag"
    # cmd = "aig"
    # cmd = "aimg"

    ## ------ 2. label stage (init: bit-level post-syn, route: bit-level post-layout) ------
    ## ------ (init_word: signal-level post-syn, route_word: signal-level post-layout) ------
    # label_cmd = "_init"
    # label_cmd = "_route"
    # label_cmd = "_init_word"
    label_cmd = "_route_word"

    global cover_all_lst
    cover_all_lst = []

    with open ("./design_js/design_lst.json", "r") as f:
        design_lst = json.load(f)
    k_fold(design_lst)

    cover_avg = round(np.mean(cover_all_lst), 2)
    print(f"Average Coverage: {cover_avg}%")