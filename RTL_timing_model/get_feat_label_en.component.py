import re, os, time, json, copy
import numpy as np
from stat_ import MAPE, regression_metrics

def autoRun(design_name, design_top):
    print('Current Design: ', design_name)
    feat_label_dir = "../preprocess/feat_label_pwr"
    with open (f"{feat_label_dir}/{design_name}_sog{label_cmd}.json", "r") as f:
        feat_label_design = json.load(f)
    feat_label_design_new = []

    for idx, module_dct in enumerate(feat_label_design):
        module_dct_new = copy.deepcopy(module_dct)
        feat_vec = module_dct['feat']
        feat_vec_reg = module_dct['feat_reg']
        feat_vec_comb = module_dct['feat_comb']
        for cmd in ['xag', 'aig', 'aimg']:
            feat_label_dir = "../preprocess/feat_label_pwr"
            with open (f"{feat_label_dir}/{design_name}_{cmd}{label_cmd}.json", "r") as f:
                feat_label_design = json.load(f)
                
            # print(len(feat))
            feat_vec.extend(feat_label_design[idx]['feat'])
            feat_vec_reg.extend(feat_label_design[idx]['feat_reg'])
            feat_vec_comb.extend(feat_label_design[idx]['feat_comb'])

        module_dct_new['feat'] = feat_vec
        module_dct_new['feat_reg'] = feat_vec_reg
        module_dct_new['feat_comb'] = feat_vec_comb
        # feat_vec = np.array(feat_vec)
        # print(feat_vec.shape)
        # exit(0)

        # print(feat_vec.shape)
        feat_label_design_new.append(module_dct_new)
    with open (f"./feat_label_en/{design_name}_en{label_cmd}.json", 'w') as f:
        json.dump(feat_label_design_new, f, indent=4)



def run_one_bench(bench, design_data, name=None):

    bench_root = design_data[bench]
    for k, v in bench_root.items():
        if name:
            if k == name:
                design_top = v[0]
                clk_name = v[1]
                # run_all_cons_for_one_design(bench, k, design_top, clk_name)
                autoRun(k, design_top)
        else:
            design_top = v[0]
            clk_name = v[1]
            # run_all_cons_for_one_design(bench, k, design_top, clk_name)
            autoRun(k, design_top)





if __name__ == '__main__':

    design_json = f"../preprocess/rtl_pwr_data/design_RTL_timer_power.json"

    global pwr_comp, pwr_tpe
    # pwr_tpe = "comb"
    pwr_tpe = "reg"
    # pwr_comp = f"{pwr_tpe}_dyn"
    pwr_comp = f"clk_dyn"


    global cmd, label_cmd
    label_cmd = "_init"
    # label_cmd = "_route"
    # label_cmd = ""

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    
    design_name = ""
    bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
    # bench_list = ['chipyard', 'riscvcores', 'NVDLA']

    global module_cnt
    module_cnt = 0
    
    for bench in bench_list:
        run_one_bench(bench, design_data, design_name)


    print(f"Total Module Count: {module_cnt}")