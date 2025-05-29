import re, os, time, json, pickle
from multiprocessing import Pool
from timing_path import timing_slack_rpt_parser

corner = "TYP"

def parse_qor_rpt(rpt_path):
    with open (rpt_path, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        setup_re = re.findall(r"^  Timing Path Group (\S+) \(max_delay/setup\)", line)
        if setup_re:
            wns = float(re.findall(r"^  Critical Path Slack:(\s+)(\S+)", lines[idx+4])[0][1])
            tns = float(re.findall(r"^  Total Negative Slack:(\s+)(\S+)", lines[idx+5])[0][1])
            path_num = int(re.findall(r"^  No. of Violating Paths:(\s+)(\S+)", lines[idx+6])[0][1])
        area_re = re.findall(r"^  Area", line)
        if area_re:
            area = int(float(re.findall(r"  Design Area:(\s+)(\S+)", lines[idx+4])[0][1]))
        cell_cnt_re = re.findall(r"^  Cell & Pin Count", line)
        if cell_cnt_re:
            pin_cnt = int(re.findall(r"^  Pin Count:(\s+)(\S+)", lines[idx+2])[0][1])
            cell_cnt = int(re.findall(r"^  Leaf Cell Count:(\s+)(\S+)", lines[idx+5])[0][1])
    print(f"wns: {wns}, tns: {tns}, path_num: {path_num}, area: {area}, cell_cnt: {cell_cnt}, pin_cnt: {pin_cnt}")

    design_dict = {
        'wns': wns,
        'tns': tns,
        'path_num': path_num,
        'area': area,
        'cell_cnt': cell_cnt,
        'pin_cnt': pin_cnt
    }

    return design_dict


def parse_slack_rpt(rpt_path):
    slack_paser = timing_slack_rpt_parser()
    slack_paser.parse_rpt(rpt_path)
    reg_feat_dct = slack_paser.reg_feat_dct
    reg_cnt = slack_paser.path_cnt
    print(f"reg_cnt: {reg_cnt}")
    return reg_feat_dct

    


def autoRun(design_name, design_top):
    print('Current Design:', design_name)
    print("Hierarchical Power Report")
    qor_rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/{design_top}.qor.rpt"
    parse_qor_rpt(qor_rpt_path)
    # exit()
    slack_rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/{design_top}.timing_slack.rpt"
    reg_feat_dct = parse_slack_rpt(slack_rpt_path)
    # print(f"reg_feat_dct: {reg_feat_dct}")
    if not os.path.exists(f"./save_rpt/bog_timing_rpt/{cmd}/"):
        os.makedirs(f"./save_rpt/bog_timing_rpt/{cmd}", exist_ok=True)
    if not os.path.exists(f"./save_rpt/bog_reg_lst/"):
        os.makedirs(f"./save_rpt/bog_reg_lst/", exist_ok=True)
    with open(f"./save_rpt/bog_timing_rpt/{cmd}/{design_name}_{cmd}.pkl", 'wb') as f:
        pickle.dump(reg_feat_dct, f)
    
    with open(f"./save_rpt/bog_reg_lst/{design_name}.json", 'w') as f:
        json.dump(list(reg_feat_dct.keys()), f, indent=4)

    
def run_one_bench(bench, design_data, name=None):

    bench_root = design_data[bench]
    for k, v in bench_root.items():
        if name:
            if k == name:
                design_top = v[0]
                # run_all_cons_for_one_design(bench, k, design_top, clk_name)
                autoRun(k, design_top)
        else:
            design_top = v[0]
            # run_all_cons_for_one_design(bench, k, design_top, clk_name)
            autoRun(k, design_top)



if __name__ == '__main__':

    design_json = f"../design_rtl_timer_pwr.json"

    global cmd

    cmd = 'sog'
    # cmd = 'aig'
    # cmd = 'aimg'
    # cmd = 'xag'
    # phase = 'route'

    assert cmd in ['sog', 'aig', 'aimg', 'xag']

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    
    design_name = "TinyRocket"
    bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
    # bench_list = ['chipyard', 'riscvcores', 'NVDLA']

    
    for bench in bench_list:
        run_one_bench(bench, design_data, design_name)

    # print(f"Total Module Count: {module_cnt}")