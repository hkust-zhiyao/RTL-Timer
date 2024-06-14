import os, time, json, pickle
from multiprocessing import Pool
import numpy as np
from eval import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good_tns.json"



def run_one_design(design_name):
    #### load data ####
    wns_feat_vec, tns_feat_vec = [],[]
    with open (f"./data/graph_data/{design_name}_rtlil_graph_feat.json", "r") as f:
        graph_feat = json.load(f)

    
    wns_feat_vec.extend(graph_feat)
    tns_feat_vec.extend(graph_feat)

    tns_lst, wns_lst = [], []
    
    for cmd_tmp in cmd_lst:
        with open (f"./data/timing_data/{design_name}_wns_{cmd_tmp}.pkl", "rb") as f:
            wns_vec = pickle.load(f)
            wns_feat_vec.extend(wns_vec)
            wns_lst.append(wns_vec[0])

        with open (f"./data/timing_data/{design_name}_tns_{cmd_tmp}.pkl", "rb") as f:
            tns_vec = pickle.load(f)
            tns_feat_vec.extend(tns_vec)
            tns_lst.append(tns_vec[0])
    
    tns_feat_vec.append(np.mean(np.array(tns_lst)))
    wns_feat_vec.append(np.mean(np.array(wns_lst)))

    
    with open (f"./data/label_data/{design_name}_tns.pkl", "rb") as f:
        tns_label = pickle.load(f)
    with open (f"./data/label_data/{design_name}_wns.pkl", "rb") as f:
        wns_label = pickle.load(f)
    print(tns_label, design_name)

    wns_feat_lst.append(wns_feat_vec)
    tns_feat_lst.append(tns_feat_vec)
    wns_label_lst.append(wns_label)
    tns_label_lst.append(tns_label)




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

    design_name = 'TinyRocket'
    design_name = ""
    global cmd, design_wns_pred_lst, design_wns_real_lst, design_tns_pred_lst, design_tns_real_lst
    cmd = 'en'
    global cmd_lst
    cmd_lst = ['sog', 'aig', 'xag', 'aimg']
    design_wns_pred_lst, design_wns_real_lst, design_tns_pred_lst, design_tns_real_lst =[],[],[],[]

    global feat_lst, wns_label_lst, tns_label_lst
    wns_feat_lst, tns_feat_lst, wns_label_lst, tns_label_lst = [], [],[],[]

    for bench in bench_list_all:
        run_all(bench, design_name)




    with open(f"./data/feat_label/feat_wns_{cmd}.pkl", "wb") as f:
        pickle.dump(wns_feat_lst, f)
    with open(f"./data/feat_label/feat_tns_{cmd}.pkl", "wb") as f:
        pickle.dump(tns_feat_lst, f)
    with open("./data/feat_label/label_wns.pkl", "wb") as f:
        pickle.dump(wns_label_lst, f)
    with open("./data/feat_label/label_tns.pkl", "wb") as f:
        pickle.dump(tns_label_lst, f)