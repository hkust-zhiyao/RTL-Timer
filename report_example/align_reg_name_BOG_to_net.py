import re, os, time, json, pickle
from multiprocessing import Pool
from timing_path import timing_slack_rpt_parser


corner = "TYP"

def change_reg_name(reg_name):
    reg_name = re.sub(r"\$_(\S+)", "", reg_name)
    reg_name = re.sub(r"/(\w+)\.", "/", reg_name)
    reg_name = re.sub(r"^_", "u_", reg_name)
    reg_name = re.sub(r"/_", "/u_", reg_name)

    ps_num_re = re.findall(r"\[(\d+)\]\[(\d+)\]", reg_name)
    if ps_num_re:
        ps_num1 = int(ps_num_re[0][0])
        ps_num2 = int(ps_num_re[0][1])
        reg_name = re.sub(r"\[(\d+)\]\[(\d+)\]", f"_reg_{ps_num1}__{ps_num2}_", reg_name)

        return reg_name

    ptr_num_re = re.findall(r"\[(\d+)\]", reg_name)
    if ptr_num_re:
        ptr_num = int(ptr_num_re[0])
        reg_name = re.sub(r"\[(\d+)\]", f"_reg_{ptr_num}_", reg_name)

        return reg_name
    
    reg_name = reg_name + '_reg'
    return reg_name


def align_reg_name(bog_dct, net_dct):
    bog_dct_new = {}
    for reg_name, val in bog_dct.items():
        reg_name = change_reg_name(reg_name)
        bog_dct_new[reg_name] = val
    
    common_reg_name_set = set(bog_dct_new.keys()).intersection(set(net_dct.keys()))
    coverage = len(common_reg_name_set) / len(net_dct.keys())
    print(f"Coverage: {coverage:.2%}")

    bog_dct_final, net_dct_final = {}, {}
    for reg_name in common_reg_name_set:
        bog_dct_final[reg_name] = bog_dct_new[reg_name]
        net_dct_final[reg_name] = net_dct[reg_name]
        print(f"Reg Name: {reg_name}")
        print(f"BOG: {bog_dct_final[reg_name]}")
        print(f"Net: {net_dct_final[reg_name]}")

    
    print(f"Orignal BOG Count: {len(bog_dct)}, Net Count: {len(net_dct)}")
    print(f"Final BOG Count: {len(bog_dct_final)}, Net Count: {len(net_dct_final)}")

    return bog_dct_final, net_dct_final


    



    
def clean_net_dct(net_dct):
    net_dct_new = {}
    for reg_name, val in net_dct.items():
        if "POWERGATING" in reg_name:
            continue
        net_dct_new[reg_name] = val
    
    return net_dct_new




def autoRun(design_name, design_top):
    print('Current Design:', design_name)

    
    net_dct_path = f"./save_rpt/net_timing_rpt_{phase}/{design_name}.json"
    with open(net_dct_path, 'r') as f:
        net_dct = json.load(f)
    net_dct = clean_net_dct(net_dct)


    # for cmd in ['sog', 'aig', 'aimg', 'xag']:
    bog_dct_path = f"./save_rpt/bog_timing_rpt/{cmd}/{design_name}_{cmd}.pkl"
    with open(bog_dct_path, 'rb') as f:
        bog_dct = pickle.load(f)

    bog_dct, net_dct = align_reg_name(bog_dct, net_dct)

    if not os.path.exists(f"./rtl_timing_data/{phase}/{cmd}/"):
        os.makedirs(f"./rtl_timing_data/{phase}/{cmd}", exist_ok=True)
    if not os.path.exists(f"./rtl_timing_data/{phase}/net/"):
        os.makedirs(f"./rtl_timing_data/{phase}/net", exist_ok=True)

    with open(f"./rtl_timing_data/{phase}/{cmd}/{design_name}_{cmd}.pkl", 'wb') as f:
        pickle.dump(bog_dct, f)
    with open(f"./rtl_timing_data/{phase}/net/{design_name}.json", 'w') as f:
        json.dump(net_dct, f, indent=4)
    

    
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
    with open(design_json, 'r') as f:
        design_data = json.load(f)

    global cmd, phase

    cmd = 'sog'
    cmd = 'aig'
    cmd = 'aimg'
    cmd = 'xag'


    phase = 'init'
    phase = 'route'


    for cmd in ['sog', 'aig', 'aimg', 'xag']:
        for phase in ['init', 'route']:
            print(f"Running for {cmd} in {phase} phase")
            assert cmd in ['sog', 'aig', 'aimg', 'xag']

            

            
            design_name = "TinyRocket"
            bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']

            
            for bench in bench_list:
                run_one_bench(bench, design_data, design_name)
