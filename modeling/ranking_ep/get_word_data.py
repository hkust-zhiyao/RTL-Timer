import os, time, json, pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
from word_stat import bit2word, avg_stat
from draw_fig import draw_fig
from eval import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

cur_dir = os.getcwd()

def run_one_design(design_name):
    print('Current Design: ', design_name)
    data_dir = f"{cur_dir}/../../data/feat/{cmd}/pair"
    graph_feat_dir = f"{cur_dir}/../../data/feat/graph_feat"
    with open (f"{data_dir}/{design_name}_{phase}_feat.pkl", "rb") as f:
        feat = pickle.load(f)
    with open (f"{data_dir}/{design_name}_{phase}_feat_dict.pkl", "rb") as f:
        feat_dict = pickle.load(f)
    with open (f"{graph_feat_dir}/{design_name}_rtlil_graph_feat.json", "r") as f:
        feat_graph = json.load(f)
    with open (f"{data_dir}/{design_name}_{phase}_label.pkl", "rb") as f:
        label = pickle.load(f)

    with open (f"{data_dir}/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    with open (f"{data_dir}/../pair_rank/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_rank_dict = pickle.load(f)


    feat = pd.DataFrame(feat)
    feat.drop(feat.columns[[25]], axis=1, inplace=True)

    with open (f"/home/coguest5/ep_prediction/ML_model/model_sog_sta_syn/saved_design_model/ep_model_{design_name}.pkl", "rb") as f:
        xgbr = pickle.load(f)

    pred = xgbr.predict(feat)

    pred_dict = {}

    idx = 0
    for ep_name, _ in label_dict.items():
        pred_dict[ep_name] = pred[idx]
        idx += 1

    
    b2w = bit2word(design_name, feat_dict)

    pred_lst, label_lst = [], []
    for ep_label, label in label_dict.items():
        vec = pred_dict[ep_label]
        # i += 1
        pred_lst.append(vec)
        label_lst.append(label)
        b2w.convert_bit_2_word(ep_label, vec, label)
    print('ll', len(pred_lst))
    r_w = b2w.get_word_stat()
    r_b = b2w.get_bit_stat()
    word_r.append(r_w)
    bit_r.append(r_b)

    b2w.save_dict("./pred", cmd)

    word_pred_dict = b2w.word_pred_dict
    word_real_dict = b2w.word_real_dict
    pred = np.array(list(word_pred_dict.values()))
    label = np.array(list(word_real_dict.values()))


    r = R_corr(pred, label)
    mape_val = MAPE(pred, label)
    bit_r.append(r)

    tau1 = kendall_tau(pred, label)
    tau2 = kendall_tau_rank(pred, label)
    tau = max(tau1, tau2)


    bit_c, word_c = coverage_word(word_pred_dict, label_rank_dict)

    bit_cover.append(bit_c)
    word_cover.append(word_c)

    stat_dict = {}
    stat_dict['R'] = r
    stat_dict['MAPE'] = mape_val
    stat_dict['Kendall Tau'] = tau
    stat_dict['Bit Coverage'] = 0
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
    cmd = "aig"
    cmd = "aimg"
    cmd = "xag"

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
    
    avg_stat(word_r, 'Word')

    avg_stat(bit_r, 'Bit')

    avg_stat(bit_cover, 'Bit Cover')

    avg_stat(word_cover, 'Word Cover')


    with open (f'./stat/stat_{cmd}_ori.json', 'w') as f:
        json.dump(saved_stat_dict, f)
    
    
