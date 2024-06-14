import os, time, json, pickle
from multiprocessing import Pool
import pandas as pd
import numpy as np
from draw_fig import draw_fig
from word_stat import bit2word, avg_stat
import matplotlib.pyplot as plt
from eval import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_good.json"

# design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_rtl_timer.json"

# design_json = "/home/coguest5/AIG_analyzer/LS-benchmark/design_timing_rgb_no_cache.json"

def draw_scatter(pred_dict, sog_dict, aig_dict, aimg_dict, xag_dict, real_dict):
    pred_lst = np.array(list(pred_dict.values()))
    real_lst = np.array(list(real_dict.values()))
    sog_lst = np.array(list(sog_dict.values()))
    aig_lst = np.array(list(aig_dict.values()))
    aimg_lst = np.array(list(aimg_dict.values()))
    xag_lst = np.array(list(xag_dict.values()))
    fig, ax = plt.subplots(figsize=(14,12), dpi=800)

    r = round(R_corr(sog_lst, real_lst),2)
    mape_val = MAPE(sog_lst, real_lst)
    ax.scatter(real_lst, sog_lst, alpha= 0.6, s=900, label=f"SOG (R = {r}, MAPE = {mape_val}%)")

    r = round(R_corr(aig_lst, real_lst),2)
    mape_val = MAPE(aig_lst, real_lst)
    ax.scatter(real_lst, aig_lst, alpha= 0.6, s=1500, marker='^', label=f"AIG (R = {r}, MAPE = {mape_val}%)")

    r = round(R_corr(aimg_lst, real_lst),2)
    mape_val = MAPE(aimg_lst, real_lst)
    ax.scatter(real_lst, aimg_lst, alpha= 0.4, s=900, marker='P', label=f"AIMG (R = {r}, MAPE = {mape_val}%)")

    r = round(R_corr(xag_lst, real_lst),2)
    mape_val = MAPE(xag_lst, real_lst)
    ax.scatter(real_lst, xag_lst, alpha= 0.2, s=900, marker='*', label=f"XAG (R = {r}, MAPE = {mape_val}%)")

    ax.scatter(real_lst, pred_lst, alpha= 1, s=1500, marker='.', label=f"En-regr (R = 0.91, MAPE = 5%)")

    ax.plot([0.300, 0.680], [0.300, 0.680], ls="--", c='b', alpha = 0.4, linewidth = 6)
    
    
    plt.legend(prop={'family':'Times New Roman', 'size':40}, loc='upper left')
    plt.xticks(fontproperties='Times New Roman', size=60)
    plt.yticks(fontproperties='Times New Roman', size=60)

    # plt.xticks(rotation=-15) 
    plt.ylim(0.200,1.99)
    plt.xlabel(f"Post-Syn Signal AT (ns)", fontdict={'family':'Times New Roman', 'size':72, 'weight':'bold'})
    plt.ylabel(f"Pred Signal AT (ns)", fontdict={'family':'Times New Roman', 'size':72, 'weight':'bold'})
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.4))
    plt.show()
    plt.savefig(f"./fig/EP_{design_name}_en.png", dpi=800, bbox_inches='tight')
    plt.close()

    
    


def run_one_design(design_name):
    print('Current Design: ', design_name)
    data_dir = "./pred/en/"
    with open (f"{data_dir}/{design_name}_feat_dict.pkl", "rb") as f:
        feat_dict = pickle.load(f)
    with open (f"{data_dir}/{design_name}_label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    feat = list(feat_dict.values())
    label = np.array(list(label_dict.values()))

    feat = pd.DataFrame(feat)
    feat.drop(feat.columns[[9,10]], axis=1, inplace=True)
    # df_label = pd.DataFrame(label_all)

    with open (f"./saved_design_model_word/{cmd}/ep_model_{design_name}.pkl", "rb") as f:
        xgbr = pickle.load(f)

    pred = xgbr.predict(feat)
    pred_dict = {}

    idx = 0
    for ep, _ in label_dict.items():
        pred_dict[ep] = pred[idx]
        idx += 1

    r = draw_fig(design_name, list(label_dict.values()), list(pred_dict.values()), cmd, "./fig")
    r_dict[design_name] = r


    r = R_corr(pred, label)
    mape_val = MAPE(pred, label)
    tau1 = kendall_tau(pred, label)
    tau2 = kendall_tau_rank(pred, label)
    tau = max(tau1, tau2)

    data_dir = f"../../data/feat/sog/pair"
    with open (f"{data_dir}/../pair_rank/{design_name}_{phase}_label_dict.pkl", "rb") as f:
        label_rank_dict = pickle.load(f)

    with open (f"./pred/sog/b18_1_word_pred.pkl", "rb") as f:
        sog_word_dict = pickle.load(f)
    with open (f"./pred/aig/b18_1_word_pred.pkl", "rb") as f:
        aig_word_dict = pickle.load(f)
    with open (f"./pred/aimg/b18_1_word_pred.pkl", "rb") as f:
        aimg_word_dict = pickle.load(f)
    with open (f"./pred/xag/b18_1_word_pred.pkl", "rb") as f:
        xag_word_dict = pickle.load(f)
    
    sog_dict, aig_dict, aimg_dict, xag_dict = {},{},{},{}
    for ep, _ in pred_dict.items():
        sog_dict[ep] = sog_word_dict[ep]
        aig_dict[ep] = aig_word_dict[ep]
        aimg_dict[ep] = aimg_word_dict[ep]
        xag_dict[ep] = xag_word_dict[ep]

    draw_scatter(pred_dict, sog_dict, aig_dict, aimg_dict, xag_dict, label_dict)

    bit_c, word_c = coverage_word(pred_dict, label_rank_dict)
    # bit_cover.append(bit_c)
    # word_cover.append(word_c)

    stat_dict = {}
    stat_dict['R'] = r
    stat_dict['MAPE'] = mape_val
    stat_dict['Kendall Tau'] = tau
    stat_dict['Bit Coverage'] = 0
    stat_dict['Word Coverage'] = word_c

    saved_stat_dict[design_name] = stat_dict

    r_lst.append(r)
    cover_lst.append(word_c)


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
    design_name = "b18_1"


    global phase, bit, tool, cmd, r_dict
    r_dict = {}
    cmd = "en"

    phase = 'PREOPT'
    phase = 'SYN'

    if phase == 'PREOPT':
        tool = 'innovus'
    else:
        tool = 'dc'
    
    global saved_stat_dict
    saved_stat_dict = {}

    global r_lst, cover_lst 
    r_lst, cover_lst = [], []


    for bench in bench_list_all:
        run_all(bench, design_name)


    avg_stat(r_lst, 'EN-word')
    avg_stat(cover_lst, 'EN-word')

        # exit()

    # with open (f'./stat/stat_{cmd}.json', 'w') as f:
    #     json.dump(saved_stat_dict, f)


