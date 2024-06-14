import os, time, json, pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
from word_stat import bit2word, avg_stat
from collections import defaultdict
from draw_fig import draw_fig
from eval import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

cur_dir = os.getcwd()

def run_one_design(design_name):
    cmd_lst = ['sog','aig','xag','aimg']
    label_dir = f"../../data/feat/sog/pair_rank/{design_name}_SYN_label_dict.pkl"
    with open (label_dir, "rb") as f:
        label_dict = pickle.load(f)
    ep_set = label_dict.keys()
    for cmd in cmd_lst:
        label_dir = f"../../data/feat/{cmd}/pair_rank/{design_name}_SYN_label_dict.pkl"
        with open (label_dir, "rb") as f:
            label_dict_cmd = pickle.load(f)
        ep_set = label_dict_cmd.keys() & ep_set
    
    label_d = {}
    for ep in ep_set:
        label_d[ep] = label_dict[ep]
    
    feat_d_tmp = defaultdict(list)
    for cmd in cmd_lst:
        feat_dir = f"./pred/{cmd}/{design_name}_pred_dict.pkl"
        with open (feat_dir, "rb") as f:
            feat_dict_cmd = pickle.load(f)
        
        for ep in ep_set:
            feat_d_tmp[ep].append(feat_dict_cmd[ep])

    graph_feat_dir = "../../data/feat/graph_feat/"
    with open (f"{graph_feat_dir}/{design_name}_rtlil_graph_feat.json", "r") as f:
        feat_graph = json.load(f)

    feat_d = {}
    for ep, feat in feat_d_tmp.items():
        vec = feat
        feat_arr = np.array(feat)
        vec.extend([np.mean(feat_arr), np.max(feat_arr), np.min(feat_arr)])
        vec.extend(feat_graph)
        feat_d[ep] = vec


    with open (f"./pred/en/{design_name}_feat_dict.pkl", "wb") as f:
        pickle.dump(feat_d, f)
    with open (f"./pred/en/{design_name}_label_dict.pkl", "wb") as f:
        pickle.dump(label_d, f)
 
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


    phase = 'PREOPT'
    phase = 'SYN'

    if phase == 'PREOPT':
        tool = 'innovus'
    else:
        tool = 'dc'
    
    for bench in bench_list_all:
        run_all(bench, design_name)
    

    
