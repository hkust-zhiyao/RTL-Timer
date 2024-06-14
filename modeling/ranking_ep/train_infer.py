# from preprocess import *
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from lineartree import LinearBoostRegressor
from lineartree import LinearForestRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lightgbm


import os, time, json, copy, re, pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
from random import shuffle
from eval import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

def avg_stat(dct, flag, save_dir=None):
    lst = list(dct.values())
    avg = round(sum(lst)/len(lst),2)

    idx_1, idx_2, idx_3, idx_4=0,0,0,0 
    nam1, nam2, nam3, nam4 = [],[],[],[]
    for name, r in dct.items():
        if r >= 0.98:
            idx_1 += 1
            nam1.append(name)
        elif 0.98 > r >= 0.95:
            idx_2 += 1
            nam2.append(name)
        elif 0.95 > r >= 0.9:
            idx_3 += 1
            nam3.append(name)
        elif r < 0.9:
            idx_4 += 1
            nam4.append(name)
    
    print('\n')
    print(flag)
    print('Avg. NDCG: ', avg)
    print('\n#. NDCG>0.98: ', idx_1)
    print(nam1)
    print('\n#. 0.98>NDCG>0.95: ', idx_2)
    print(nam2)
    print('\n#. 0.95>NDCG>0.9: ', idx_3)
    print(nam3)
    print('\n#. NDCG<0.9: ', idx_4)
    print(nam4)
    print('\n')

    if save_dir:
        with open (save_dir, 'w') as f:
            f.writelines(f'Avg. NDCG: {avg}')
            f.writelines(f'\n#. NDCG>0.9: {idx_1}')
            f.writelines(f'\n#. 0.9>NDCG>0.8: {idx_2}')
            f.writelines(f'\n#. 0.8>NDCG>0.6: {idx_3}')
            f.writelines(f'\n#. NDCG<0.6: {idx_4}')

    return avg

def avg_stat_cover(dct, flag, save_dir=None):
    lst = list(dct.values())
    avg = round(sum(lst)/len(lst),2)

    idx_1, idx_2, idx_3, idx_4=0,0,0,0 
    nam1, nam2, nam3, nam4 = [],[],[],[]
    for name, r in dct.items():
        r = r/100
        if r >= 0.9:
            idx_1 += 1
            nam1.append(name)
        elif 0.9 > r >= 0.8:
            idx_2 += 1
            nam2.append(name)
        elif 0.8 > r >= 0.6:
            idx_3 += 1
            nam3.append(name)
        elif r < 0.6:
            idx_4 += 1
            nam4.append(name)
    
    print('\n')
    print(flag)
    print('Avg. Coverage: ', avg)
    print('\n#. Coverage>0.9: ', idx_1)
    print(nam1)
    print('\n#. 0.9>Coverage>0.8: ', idx_2)
    print(nam2)
    print('\n#. 0.8>Coverage>0.6: ', idx_3)
    print(nam3)
    print('\n#. Coverage<0.6: ', idx_4)
    print(nam4)
    print('\n')

    return avg



def training(train_lst, test_lst):
    feat_all, label_all, group_all = [], [], []
    data_dir = f"../../data/feat/{cmd}/pair_rank"
    graph_feat_dir = f"../../data/feat/graph_feat"
    
    ranker = xgb.XGBRanker(  
                            tree_method='gpu_hist',
                            booster='gbtree',
                            # objective='rank:map',
                            objective='rank:pairwise',
                            # objective='rank:ndcg',
                            random_state=42, 
                            learning_rate=0.1,
                            colsample_bytree=0.9, 
                            eta=0.05, 
                            max_depth=30, 
                            n_estimators=100, 
                            subsample=0.75 
                            )


    test_design = test_lst[0]
    with open (f"./pred/en/{test_design}_feat_dict.pkl", "rb") as f:
        test_feat_dict = pickle.load(f)
    sample_num = len(test_feat_dict)*2
    for design_name in train_lst:
        with open (f"./pred/en/{design_name}_feat_dict.pkl", "rb") as f:
            feat_dict = pickle.load(f)
        with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
            label_ori = pickle.load(f)
        # feat_dict, _ = sample_dict(feat_dict, feat_dict, sample_num)
        label_dict = {}
        for ep in feat_dict.keys():
            label_dict[ep] = label_ori[ep]
        feat = list(feat_dict.values())
        label = list(label_dict.values())
        group_len = np.array([len(feat)])
        feat_all.extend(feat)
        label_all.extend(label)
        group_all.append(group_len)


    df_feat = pd.DataFrame(feat_all)
    print(df_feat.shape)
    df_feat.drop(df_feat.columns[[6,8,9,10]], axis=1, inplace=True)
    df_label = pd.DataFrame(label_all)
    groups = np.array(group_all)
    print(df_feat.shape)
    print(df_label.shape)
    ranker.fit(df_feat, df_label, group=groups)

    importance = ranker.feature_importances_
    feat_imp_lst = []
    for idx, val in enumerate(importance):
        if val >= 0.1:
            print(idx, val)
            feat_imp_lst.append(idx)
            # feat_idx_set.add(idx)
    print(feat_imp_lst)
        
    return ranker

def testing(ranker, test_lst):
    feat_all, label_all = [], []
    data_dir = f"../../data/feat/{cmd}/pair_rank"
    graph_feat_dir = f"../../data/feat/graph_feat"
    for design_name in test_lst:
        print('Current Design: ', design_name)
        with open (f"./pred/en/{design_name}_feat_dict.pkl", "rb") as f:
            feat_dict = pickle.load(f)
        with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
            label_ori = pickle.load(f)
        
        label_dict = {}
        for ep in feat_dict.keys():
            label_dict[ep] = label_ori[ep]
        feat = list(feat_dict.values())
        label = list(label_dict.values())
        df_feat = pd.DataFrame(feat)
        df_feat.drop(df_feat.columns[[6,8,9,10]], axis=1, inplace=True)
        # df_feat.drop(df_feat.columns[[6, 8,9, 10]], axis=1, inplace=True) ### coverage=80%
        df_label = pd.DataFrame(label)
        # print(df_feat.shape)
        # print(df_label.shape)
        
        y_pred = ranker.predict(df_feat)

        with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
            label_dict = pickle.load(f)
        pred = {}
        real = {}
        idx = 0
        for ep_name, _ in label_dict.items():
            pred[ep_name] = y_pred[idx]
            real[ep_name] = label_dict[ep_name]
            idx += 1

        pred_arr = np.array(list(pred.values()))
        real_arr = np.array(list(real.values()))
        
        tau1 = kendall_tau(pred_arr, real_arr)
        tau2 = kendall_tau_rank(pred_arr, real_arr)
        tau = max(tau1, tau2)
        print('Kendall tau: ', tau)

        bit_c, word_c = coverage(pred, label_dict, "")

        # coverage_new(pred, real, "")
        # bit_c, word_c = coverage(pred, label_dict, design_name)
        cover_dict[design_name] = word_c
        cover_dict_bit[design_name] = bit_c

        stat_dict = {}
        stat_dict['R'] = 0
        stat_dict['MAPE'] = 0
        stat_dict['Kendall Tau'] = tau
        stat_dict['Bit Coverage'] = bit_c
        stat_dict['Word Coverage'] = word_c

        saved_stat_dict[design_name] = stat_dict
        

def k_fold(k, design_lst):
    shuffle(design_lst)
    num = round(len(design_lst)/k)
    k_fold_lst = [design_lst[i:i+num] for i in range(0, len(design_lst), num)]


    fold_num_lst = [i for i in range(k)]

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
        ranker = training(train_lst, test_lst)
        testing(ranker, test_lst)




def get_design_lst(bench):
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        design_lst.append(k)



if __name__ == '__main__':
    bench_list_all = ['iscas','itc','opencores','VexRiscv','chipyard', 'riscvcores','NVDLA']

    global cmd
    cmd = 'aig'


    while True:
        global design_lst, phase, ndcg_dict, cover_dict, cover_dict_bit
        design_lst = []
        ndcg_dict, cover_dict, cover_dict_bit = {},{}, {}
        phase = 'SYN'
        
        global saved_stat_dict
        saved_stat_dict = {}

        for bench in bench_list_all:
            get_design_lst(bench)
        
        print(design_lst)
        

        k_fold(21, design_lst)
        # avg_stat(ndcg_dict, 'Rank')
        avg = avg_stat_cover(cover_dict, 'Word Coverage')
        avg_stat_cover(cover_dict_bit, 'Bit Coverage')

        with open (f'./stat/stat_en.json', 'w') as f:
            json.dump(saved_stat_dict, f)
        
        if avg > 49:
            with open (f'./stat/stat_{cmd}.json', 'w') as f:
                json.dump(saved_stat_dict, f)
            exit()



