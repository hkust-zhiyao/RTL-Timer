import re, os, time, json, pickle
from stat_ import MAPE, regression_metrics



def reg_bit_to_word(reg_dct_bit, is_net=False):
    reg_dct_word = {}
    for reg_name, val in reg_dct_bit.items():
        reg_name_word = re.sub(r'_reg_(\d+)__(\d+)_', '', reg_name)
        reg_name_word = re.sub(r'_reg_(\d+)_', '', reg_name_word)
        reg_name_word = re.sub(r'_reg$', '', reg_name_word)
        if reg_name_word not in reg_dct_word:
            reg_dct_word[reg_name_word] = []
            reg_dct_word[reg_name_word].append(val)
        else:
            reg_dct_word[reg_name_word].append(val)

    print(len(reg_dct_bit), len(reg_dct_word))
    
    reg_dct_word_final = {}
    if is_net:
        ## average the value of the same register
        for reg_name, val in reg_dct_word.items():
            reg_dct_word_final[reg_name] = sum(val) / len(val)
    else:
        ## average the feature vector of the same register
        for reg_name, val in reg_dct_word.items():
            reg_dct_word_final[reg_name] = [sum(x) / len(val) for x in zip(*val)]
        

    return reg_dct_word_final
    

def autoRun(design_name, design_top):
    print('Current Design: ', design_name)

    ### design-level feature
    bog_rpt_pwr_path = f"./rtl_pwr_data/bog_pwr_rpt/{cmd}/{design_name}_{cmd}.json"
    with open(bog_rpt_pwr_path, 'r') as f:
        bog_dct_pwr = json.load(f)
    bog_feat_dct = bog_dct_pwr['module']
    design_feat = bog_feat_dct[design_top]['cell'][0:7]
    design_feat.append(bog_feat_dct[design_top]['net'][0])

    ### path-level timing feature
    bog_rpt_path = f"./rtl_timing_data/{label_cmd}/{cmd}/{design_name}_{cmd}.pkl"
    with open(bog_rpt_path, 'rb') as f:
        bog_dct_all = pickle.load(f)
    net_rpt_path = f"./rtl_timing_data/{label_cmd}/net/{design_name}.json"
    with open(net_rpt_path, 'r') as f:
        net_dct_all = json.load(f)
    
    
    bog_dct_word = reg_bit_to_word(bog_dct_all, is_net=False)
    net_dct_word = reg_bit_to_word(net_dct_all, is_net=True)

    if not os.path.exists(f"./rtl_timing_data/{label_cmd}_word/{cmd}"):
        os.makedirs(f"./rtl_timing_data/{label_cmd}_word/{cmd}")
    with open (f"./rtl_timing_data/{label_cmd}_word/{cmd}/{design_name}_{cmd}.pkl", 'wb') as f:
        pickle.dump(bog_dct_word, f)

    if not os.path.exists(f"./rtl_timing_data/{label_cmd}_word/net"):
        os.makedirs(f"./rtl_timing_data/{label_cmd}_word/net")
    with open (f"./rtl_timing_data/{label_cmd}_word/net/{design_name}.json", 'w') as f:
        json.dump(net_dct_word, f, indent=4)
    

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

    design_json = f"../design_rtl_timer_pwr.json"

    global phase, cmd, label_cmd
    cmd = 'sog'
    # cmd = 'xag'
    # cmd = 'aig'
    # cmd = 'aimg'
    label_cmd = "init"
    # label_cmd = "route"

    phase = 'SYN'
    assert phase in ['SYN', 'PREOPT', 'PLACE', 'CTS', 'ROUTE']

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    
    design_name = "TinyRocket"
    bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
    # bench_list = ['chipyard', 'riscvcores', 'NVDLA']

    global module_cnt
    module_cnt = 0
    
    for bench in bench_list:
        run_one_bench(bench, design_data, design_name)


    print(f"Total Module Count: {module_cnt}")