import os, time, json, pickle
from multiprocessing import Pool
import numpy as np
from eval import *
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

def cal_timing(delay_lst):
    ppa_dict = "/home/coguest5/AST_analyzer/std_PPA.json"
    with open(ppa_dict, 'r') as f:
        std_data = json.load(f)
    t_data = std_data['timing']
    freq = t_data['freq']
    clk_unc = t_data['clk_unc']
    lib_setup = t_data['lib_setup']
    input_delay = t_data['input_delay']
    output_delay = t_data['output_delay']
    require_time = 1/freq - clk_unc - lib_setup

    delay_sum = np.array(delay_lst)
    input_delay_array = np.zeros(delay_sum.shape)
    output_delay_array = np.zeros(delay_sum.shape)
    require_time_array = np.ones(delay_sum.shape)*require_time - output_delay_array
    arrival_time_array = delay_sum + input_delay_array
    slack_array = require_time_array - arrival_time_array
    slack_array[slack_array>0] = 0

    slack_array = np.sort(slack_array)

    tns = np.sum(slack_array)
    wns = np.min(slack_array)
    wns1 = slack_array[round(len(slack_array)*0.1)]
    wns2 = slack_array[round(len(slack_array)*0.2)]
    wns3 = slack_array[round(len(slack_array)*0.5)]
    wns4 = slack_array[round(len(slack_array)*0.8)]
    wns5 = slack_array[-1]

    print('tns:', tns)
    print('wns:', wns)
    
    return [tns, wns, wns1, wns2, wns3, wns4, wns5]


def run_one_design(design_name):
    print('Current Design: ', design_name)
    save_dir = "/home/coguest5/ep_modeling/model/tree_word/pred"
    with open (f"{save_dir}/{cmd}/{design_name}_bit_feat.pkl", "rb") as f:
        bit_feat_dict = pickle.load(f)
    with open (f"{save_dir}/{cmd}/{design_name}_bit_pred.pkl", "rb") as f:
        bit_pred_dict = pickle.load(f)
    with open (f"{save_dir}/{cmd}/{design_name}_bit_label.pkl", "rb") as f:
        bit_real_dict = pickle.load(f)
    with open (f"{save_dir}/{cmd}/{design_name}_word_pred.pkl", "rb") as f:
        word_pred_dict = pickle.load(f)
    with open (f"{save_dir}/{cmd}/{design_name}_word_label.pkl", "rb") as f:
        word_real_dict = pickle.load(f)
    with open (f"{save_dir}/{cmd}/{design_name}_b2w_map.pkl", "rb") as f:
        b2w_map = pickle.load(f)

    

    pred_lst, real_lst = [],[]
    for ep, pred in bit_pred_dict.items():
        real = bit_real_dict[ep]
        pred_lst.append(pred)
        real_lst.append(real)
    
    pred = cal_timing(pred_lst)
    real = cal_timing(real_lst)

    design_wns_pred_lst.append(pred[1])
    design_wns_real_lst.append(real[1])
    design_tns_pred_lst.append(pred[0])
    design_tns_real_lst.append(real[0])

    #### save data ####
    with open (f"./data/timing_data/{design_name}_wns_{cmd}.pkl", "wb") as f:
        pickle.dump(pred[1:], f)
    with open (f"./data/timing_data/{design_name}_tns_{cmd}.pkl", "wb") as f:
        pickle.dump([pred[0]], f)
    
    if cmd == 'sog':
        with open (f"./data/label_data/{design_name}_tns.pkl", "wb") as f:
            pickle.dump(real[0], f)
        with open (f"./data/label_data/{design_name}_wns.pkl", "wb") as f:
            pickle.dump(real[1], f)
    





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

    design_name = 'Rocket'
    design_name = ""
    global cmd, design_wns_pred_lst, design_wns_real_lst, design_tns_pred_lst, design_tns_real_lst
    cmd = 'sog'
    cmd = 'aig'
    cmd = 'xag'
    cmd = 'aimg'
    design_wns_pred_lst, design_wns_real_lst, design_tns_pred_lst, design_tns_real_lst =[],[],[],[]

    global feat_lst, wns_label_lst, tns_label_lst
    feat_lst, wns_label_lst, tns_label_lst = [], [],[]

    for bench in bench_list_all:
        run_all(bench, design_name)

    get_metrics(np.array(design_tns_pred_lst), np.array(design_tns_real_lst))
    get_metrics(np.array(design_wns_pred_lst), np.array(design_wns_real_lst))

    # if cmd == 'sog':
    #     with open(f"./data/feat_{cmd}.pkl", "wb") as f:
    #         pickle.dump(feat_lst, f)
    #     with open("./data/label_wns.pkl", "wb") as f:
    #         pickle.dump(wns_label_lst, f)
    #     with open("./data/label_tns.pkl", "wb") as f:
    #         pickle.dump(tns_label_lst, f)