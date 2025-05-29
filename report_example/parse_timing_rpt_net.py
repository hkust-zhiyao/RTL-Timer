import re, os, time, json
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
    print(f"wns: {wns}, tns: {tns}, path_num: {path_num}, area: {area}")

    design_dict = {
        'wns': wns,
        'tns': tns,
        'path_num': path_num,
        'area': area
    }

    return design_dict


def parse_slack_rpt(rpt_path):
    slack_paser = timing_slack_rpt_parser()
    slack_paser.parse_rpt(rpt_path)
    reg_slack_dct = slack_paser.reg_slack_dct
    reg_cnt = slack_paser.path_cnt
    print(f"reg_cnt: {reg_cnt}")
    return reg_slack_dct

    


def autoRun(design_name, design_top):
    print('Current Design:', design_name)
    print("Hierarchical Power Report")
    qor_rpt_path = f"./rpt_data/net/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/{design_top}.qor.rpt"
    parse_qor_rpt(qor_rpt_path)
    slack_rpt_path = f"./rpt_data/net/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/{design_top}.timing_slack.rpt"
    reg_slack_dct = parse_slack_rpt(slack_rpt_path)
    if not os.path.exists(f"./save_rpt/net_timing_rpt_{phase}/"):
        os.makedirs(f"./save_rpt/net_timing_rpt_{phase}", exist_ok=True)
    with open(f"./save_rpt/net_timing_rpt_{phase}/{design_name}.json", 'w') as f:
        json.dump(reg_slack_dct, f, indent=4)
    

    
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

    global phase

    phase = 'init'
    phase = 'route'

    assert phase in ['init', 'route']

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    
    design_name = "TinyRocket"
    bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
    # bench_list = ['chipyard', 'riscvcores', 'NVDLA']

    global path_cnt
    path_cnt = 0
    
    for bench in bench_list:
        run_one_bench(bench, design_data, design_name)

    # print(f"Total Module Count: {module_cnt}")