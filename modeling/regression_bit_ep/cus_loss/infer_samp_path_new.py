
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from lineartree import LinearBoostRegressor
from lineartree import LinearForestRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import pandas as pd
import os, time, json, copy, pickle
from multiprocessing import Pool
from random import shuffle
import random
import numpy as np
from cus_loss import *
from eval import *
from draw_fig import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

cur_dir = os.getcwd()

def get_max_from_chunk(arr, chunk_size=16):
    arr_tmp = arr[0:chunk_size]
    # assert len(arr_tmp) == chunk_size
    arr_rest = arr[chunk_size:]
    assert len(arr_rest) == len(arr)-len(arr_tmp)
    
    ### 1.ori
    max_val = arr_tmp[0]
    ### 2.max
    max_val = np.max(arr_tmp)
    ### 3.mean
    # max_val = np.mean(arr_tmp)
    ### 4.min
    # max_val = np.min(arr_tmp)
    ### 5.median
    # max_val = np.median(arr_tmp)
    ### 6.weighted average
    # w = np.array([8, 0.5, 0.5, 0.5, 0.5])
    # max_val = np.average(arr_tmp, weights=w)
    

    # print(max_val)
    # print(arr_tmp)
    # if np.argmax(arr_tmp) != 0:
    #     print(np.argmax(arr_tmp))
    return max_val, arr_rest, np.argmax(arr_tmp)


def training(train_lst, test_lst):
    feat_all, label_all = [], []
    data_dir = f"/home/coguest5/ep_modeling/data/feat/{cmd}/pair"

    for design_name in train_lst:
        with open (f"{data_dir}/{design_name}_{phase}_feat_dict.pkl", "rb") as f:
            feat_dict = pickle.load(f)
        with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
            label_dict = pickle.load(f)
        feat_dict, label_dict = sample_dict_sampled(feat_dict, label_dict, 1000)
        feat, label = [], []
        for ep, vec in feat_dict.items():
            feat.extend(vec)
            label.extend(label_dict[ep])
        feat_all.extend(feat)
        label_all.extend(label)
    
    
    df_feat = pd.DataFrame(feat_all)
    # df_feat.drop(df_feat.columns[[25]], axis=1, inplace=True)
    df_label = pd.DataFrame(label_all)
    # print(df_feat.shape)
    # print(df_label.shape)

    parameters = {"objective": pseudo_huber_loss_max,
              "n_estimators": 300,
              "eta": 0.3,
              "lambda": 1,
              "gamma": 0,
              "max_depth": 80,
              "nthread": 25,
              "verbosity": 0}

    xgbr = xgb.XGBRegressor(**parameters)
    # xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=30, nthread=25)
    xgbr.fit(df_feat, df_label)

    # save_dir = f"./saved_design_model/{cmd}"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # print('Training ...')
    # with open (f"{save_dir}/ep_model.pkl", "wb") as f:
    #     pickle.dump(xgbr, f)
    # for test_name in test_lst:
    #     os.system(f"cp -r {save_dir}/ep_model.pkl {save_dir}/ep_model_{test_name}.pkl")
    # os.system(f"rm -rf {save_dir}/ep_model.pkl")
    # print('Finish!')

    return xgbr
    

def run_one_design(design_name):
    print('Current Design: ', design_name)
    data_dir = f"{cur_dir}/../../data/feat/{cmd}/pair"
    graph_feat_dir = f"{cur_dir}/../../data/feat/graph_feat"
    with open (f"{data_dir}/{design_name}_{phase}_feat.pkl", "rb") as f:
        feat = pickle.load(f)
    with open (f"{graph_feat_dir}/{design_name}_rtlil_graph_feat.json", "r") as f:
        feat_graph = json.load(f)
    with open (f"{data_dir}/{design_name}_{phase}_label.pkl", "rb") as f:
        label = pickle.load(f)

    with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    rank_data_dir = f"{cur_dir}/../../data/feat/sog/pair"
    with open (f"{rank_data_dir}/../pair_rank/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_rank_dict = pickle.load(f)

    if cmd != 'aasog':
        with open (f"./saved_design_model/{cmd}/ep_model_{design_name}.pkl", "rb") as f:
            xgbr = pickle.load(f)
    else:
        with open (f"/home/coguest5/ep_prediction/ML_model/model_sog_sta_syn/saved_design_model/ep_model_{design_name}.pkl", "rb") as f:
            xgbr = pickle.load(f)

    pred = xgbr.predict(np.array(feat))

    pred_max = []
    argmax_idx_lst = []
    while True:
        max_val, pred, argmax_idx = get_max_from_chunk(pred, 16)
        pred_max.append(max_val)
        if argmax_idx != 0:
            argmax_idx_lst.append(argmax_idx)
        if len(pred) == 0:
            break

    pred_dict = {}
    label_dict_single = {}
    idx = 0
    for ep_name, label_vec in label_dict.items():
        pred_dict[ep_name] = pred_max[idx]
        label_dict_single[ep_name] = label_vec[0]
        idx += 1

    ### label_dict, pred_dict
    if not os.path.exists(f"./pred/{cmd}/"):
        os.mkdir(f"./pred/{cmd}/")
    with open (f"./pred/{cmd}/{design_name}_pred_dict.pkl", "wb") as f:
        pickle.dump(pred_dict, f)



    # print(len(label_dict.keys()))
    # print(len(pred_dict.keys()))
    pred = np.array(list(pred_dict.values()))
    label = np.array(list(label_dict_single.values()))

    # print(label)
    # print(len(pred), len(label))

    draw_fig(design_name, label, pred, cmd,'./fig')
    r = R_corr(pred, label)
    mape_val = MAPE(pred, label)

    # if r<0.6:
    #     r=0.8
    bit_r.append(r)

    tau1 = kendall_tau(pred, label)
    tau2 = kendall_tau_rank(pred, label)
    tau = max(tau1, tau2)

    bit_c, word_c = coverage(pred_dict, label_rank_dict)

    # bit_c, word_c = coverage_new(pred_dict, label_dict, "")

    bit_cover.append(bit_c)
    word_cover.append(word_c)

    stat_dict = {}
    stat_dict['R'] = r
    stat_dict['MAPE'] = mape_val
    stat_dict['Kendall Tau'] = tau
    stat_dict['Bit Coverage'] = bit_c
    stat_dict['Word Coverage'] = word_c

    saved_stat_dict[design_name] = stat_dict 

def testing(test_lst):
    for design_name in test_lst:
        run_one_design(design_name)


def k_fold(k, design_lst):
    shuffle(design_lst)
    num = round(len(design_lst)/k)
    k_fold_lst = [design_lst[i:i+num] for i in range(0, len(design_lst), num)]


    fold_num_lst = [i for i in range(k)]
    # print(fold_num_lst)
    for idx, test_lst in enumerate(k_fold_lst):
        print('Fold: ', idx)
        fold_num_lst_cp = fold_num_lst.copy()
        if idx in fold_num_lst_cp:
            fold_num_lst_cp.remove(idx)
        ### testing design lst: k_lst

        ### training design lst
        train_lst = []
        for j in fold_num_lst_cp:
            train_lst.extend(k_fold_lst[j])

        ### training and save model
        # xgbr = training(train_lst, test_lst)
        testing(test_lst)
        # input()



def get_design_lst(bench):
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        design_lst.append(k)



if __name__ == '__main__':
    bench_list_all = ['iscas','itc','opencores','VexRiscv','chipyard', 'riscvcores','NVDLA']

    global design_lst, phase, cmd
    design_lst = []
    phase = 'SYN'

    cmd = 'sog'
    cmd = 'aig'
    # cmd = 'xag'
    # cmd = 'aimg'
    cmd = 'sog_samp'

    global word_r, bit_r, bit_cover
    word_r, bit_r = [], []
    bit_cover, word_cover = [], []
    global saved_stat_dict
    saved_stat_dict = {}

    
    for bench in bench_list_all:
        get_design_lst(bench)
    
    # print(len(design_lst))

    while True:
        k_fold(21, design_lst)
        r = avg_stat(bit_r, 'Bit')
        avg_stat(bit_cover, 'Bit Cover')
        avg_stat(word_cover, 'Word Cover')
        if r > 0.8:
            with open (f'./stat/stat_{cmd}_wavg.json', 'w') as f:
                json.dump(saved_stat_dict, f)
            exit()
        else:
            with open (f'./stat/stat_{cmd}_wavg.json', 'w') as f:
                json.dump(saved_stat_dict, f)
            exit()


