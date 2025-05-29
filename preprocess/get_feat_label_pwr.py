import re, os, time, json
from stat_ import MAPE, regression_metrics

def autoRun(design_name, design_top):
    print('Current Design: ', design_name)
    bog_rpt_path = f"./rtl_pwr_data/bog_pwr_rpt/{cmd}/{design_name}_{cmd}.json"
    with open(bog_rpt_path, 'r') as f:
        bog_dct_all = json.load(f)
    net_rpt_path = f"./rtl_pwr_data/net_pwr_rpt{label_cmd}/{design_name}.json"
    with open(net_rpt_path, 'r') as f:
        net_dct_all = json.load(f)
    
    bog_dct = bog_dct_all['hierarch']
    bog_feat_dct = bog_dct_all['module']
    net_dct = net_dct_all['hierarch']
    net_module_dct = net_dct_all['module']

    bog_pwr_lst, net_pwr_lst = [], []

    save_lst = []

    ## get design-level feature
    design_feat = bog_feat_dct[design_top]['cell'][0:7]
    design_feat.append(bog_feat_dct[design_top]['net'][0])

    for module, bog_pwr in bog_dct.items():
        if module == design_top:
            continue
        net_pwr = net_dct[module]
        bog_feat_all = bog_feat_dct[module]['cell']
        bog_feat_all.extend(bog_feat_dct[module]['net'])
        bog_feat_all.extend(design_feat)


        bog_feat_reg = bog_feat_dct[module]['cell_reg']
        bog_feat_reg.extend(bog_feat_dct[module]['net_reg'])
        bog_feat_reg.extend(design_feat)

        bog_feat_comb = bog_feat_dct[module]['cell_comb']
        bog_feat_comb.extend(bog_feat_dct[module]['net_comb'])
        bog_feat_comb.extend(design_feat)

        bog_total_pwr = bog_pwr['total_pwr']
        net_total_pwr = net_pwr['total_pwr']


        if bog_total_pwr == 0:
            continue
        if net_total_pwr == 0:
            continue

        bog_pwr_lst.append(bog_total_pwr)
        net_pwr_lst.append(net_total_pwr)
        # save_dct = {}
        # save_dct[module] = {
        #     "feat": bog_feat,
        #     "bog_pwr": bog_total_pwr,
        #     "label": net_total_pwr
        # }
        save_dct = {
            'name': module,
            'feat': bog_feat_all,
            'feat_reg': bog_feat_reg,
            'feat_comb': bog_feat_comb,

            'bog_pwr': bog_total_pwr,
            'bog_clk_dyn': bog_feat_dct[module]['group']['clock_network']['inter_pwr'] + bog_feat_dct[module]['group']['clock_network']['switch_pwr'],
            'bog_clk_stat': bog_feat_dct[module]['group']['clock_network']['leak_pwr'],
            'bog_reg_dyn': bog_feat_dct[module]['group']['register']['inter_pwr'] + bog_feat_dct[module]['group']['register']['switch_pwr'],
            'bog_reg_stat': bog_feat_dct[module]['group']['register']['leak_pwr'],
            'bog_comb_dyn': bog_feat_dct[module]['group']['combinational']['inter_pwr'] + bog_feat_dct[module]['group']['combinational']['switch_pwr'],
            'bog_comb_stat': bog_feat_dct[module]['group']['combinational']['leak_pwr'],

            'label': net_total_pwr,
            "label_clk_dyn": net_module_dct[module]['group']['clock_network']['inter_pwr'] + net_module_dct[module]['group']['clock_network']['switch_pwr'],
            "label_clk_stat": net_module_dct[module]['group']['clock_network']['leak_pwr'],
            "label_reg_dyn": net_module_dct[module]['group']['register']['inter_pwr'] + net_module_dct[module]['group']['register']['switch_pwr'],
            "label_reg_stat": net_module_dct[module]['group']['register']['leak_pwr'],
            "label_comb_dyn": net_module_dct[module]['group']['combinational']['inter_pwr'] + net_module_dct[module]['group']['combinational']['switch_pwr'],
            "label_comb_stat": net_module_dct[module]['group']['combinational']['leak_pwr'],
        }
        save_lst.append(save_dct)
    
    with open (f"./feat_label_pwr/{design_name}_{cmd}{label_cmd}.json", 'w') as f:
        json.dump(save_lst, f, indent=4)
    
    if len(bog_pwr_lst) <= 1:
        print(f"Module Count: {len(bog_pwr_lst)}")
        return
    
    regression_metrics(bog_pwr_lst, net_pwr_lst)




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
    label_cmd = "_init"
    # label_cmd = "_route"
    label_cmd = "_route_calibre"

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