import os, time, json, pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
from word_stat import bit2word, avg_stat
from draw_fig import draw_fig
from eval import *
from cus_loss import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

cur_dir = os.getcwd()

def get_max_from_chunk(arr, chunk_size=5):
    arr_tmp = arr[0:chunk_size]
    assert len(arr_tmp) == 5
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
    with open (f"{data_dir}/../pair_rank/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_rank_dict = pickle.load(f)


    # feat = pd.DataFrame(feat)
    # print(feat.shape)
    # feat.drop(feat.columns[[25]], axis=1, inplace=True)

    if cmd != 'aasog':
        with open (f"./saved_design_model/{cmd}/ep_model_{design_name}.pkl", "rb") as f:
            xgbr = pickle.load(f)
    else:
        with open (f"/home/coguest5/ep_prediction/ML_model/model_sog_sta_syn/saved_design_model/ep_model_{design_name}.pkl", "rb") as f:
            xgbr = pickle.load(f)

    # pred = xgbr.predict(feat)


    pred = xgbr.predict(np.array(feat))

    pred_max = []
    argmax_idx_lst = []
    while True:
        max_val, pred, argmax_idx = get_max_from_chunk(pred, 5)
        pred_max.append(max_val)
        if argmax_idx != 0:
            argmax_idx_lst.append(argmax_idx)
        if len(pred) == 0:
            break

    print('Max Path Percent: ', round(len(argmax_idx_lst)/len(pred_max),2)*100)
    # print(len(pred_max), len(label_dict))
    assert len(pred_max) == len(label_dict)
    # exit()

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



    print(len(label_dict.keys()))
    print(len(pred_dict.keys()))
    pred = np.array(list(pred_dict.values()))
    label = np.array(list(label_dict_single.values()))

    # print(label)
    # print(len(pred), len(label))

    draw_fig(design_name, label, pred, cmd,'./fig')
    r = R_corr(pred, label)
    mape_val = MAPE(pred, label)
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
    








def run_all(bench, design_name=None):
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        if design_name:
            if k == design_name:
                run_one_design(k)
        else:
            run_one_design(k)



if __name__ == '__main__':
    bench_list_all = ['iscas','itc','opencores','VexRiscv','chipyard', 'riscvcores','NVDLA']

    design_name = 'VexRiscvLinuxBalanced'
    design_name = ""


    global phase, bit, tool, cmd
    cmd = "sog"
    # cmd = "aig"
    # cmd = "xag"
    # cmd = "aimg"
    
    cmd = "sog_samp"

    phase = 'PREOPT'
    phase = 'SYN'

    if phase == 'PREOPT':
        tool = 'innovus'
    else:
        tool = 'dc'
    

    global word_r, bit_r, bit_cover
    word_r, bit_r = [], []
    bit_cover, word_cover = [], []


    global saved_stat_dict
    saved_stat_dict = {}
    
    for bench in bench_list_all:
        run_all(bench, design_name)
    
    # avg_stat(word_r, 'Word')

    avg_stat(bit_r, 'Bit')

    avg_stat(bit_cover, 'Bit Cover')

    avg_stat(word_cover, 'Word Cover')


    with open (f'./stat/stat_{cmd}_wavg.json', 'w') as f:
        json.dump(saved_stat_dict, f)
    
    
