import re, os, time, json, pickle
from stat_ import MAPE, regression_metrics

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
    
    save_lst = []
    ## sort the net_dct_all by the slack_value of the dictionary {"reg_name1": slack_value1, "reg_name2": slack_value2}
    net_dct_all = dict(sorted(net_dct_all.items(), key=lambda item: item[1], reverse=True))
    for reg_name, slack_val in net_dct_all.items():
        feat_label_dct = {
            'name': reg_name,
            'feat_design': design_feat,
            'feat_path': bog_dct_all[reg_name],
            'bog_slack': bog_dct_all[reg_name][0],
            'label_slack': slack_val,
        }
        save_lst.append(feat_label_dct)
    
    with open (f"./feat_label_timing/{design_name}_{cmd}_{label_cmd}.pkl", 'wb') as f:
        pickle.dump(save_lst, f)
    
    # if len(bog_pwr_lst) <= 1:
    #     print(f"Module Count: {len(bog_pwr_lst)}")
    #     return
    
    # regression_metrics(bog_pwr_lst, net_pwr_lst)




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
    # cmd = 'sog'
    cmd = 'xag'
    # cmd = 'aig'
    # cmd = 'aimg'

    # label_cmd = "init"
    # label_cmd = "route"
    
    label_cmd = "init_word"
    # label_cmd = "route_word"

    phase = 'SYN'
    assert phase in ['SYN', 'PREOPT', 'PLACE', 'CTS', 'ROUTE']

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