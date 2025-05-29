import re, os, time, json
from multiprocessing import Pool


corner = "TYP"


def parse_hierarch_pwr(design_top, pwr_rpt_path):
    ## ================= 1. Total Power Report (hierarchical) =================
    pwr_dict = {}
    with open (pwr_rpt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_re_top = re.findall(f"{design_top}(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)", line)
            if line_re_top:
                pwr_dict[design_top] = {
                    "inter_pwr": float(line_re_top[0][1])*1000,
                    "switch_pwr": float(line_re_top[0][3])*1000,
                    "leak_pwr": float(line_re_top[0][5])*1000,
                    "total_pwr": float(line_re_top[0][7])*1000,
                    "percent": float(line_re_top[0][9]),
                }
            line_re_module = re.findall(f"  (\S+)(\s+)\((\S+)\)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)", line)
            if line_re_module:
                module_name = line_re_module[0][0]
                if "POWERGATING" in module_name:
                    continue    
                pwr_dict[module_name] = {
                    "inter_pwr": float(line_re_module[0][4])*1000,
                    "switch_pwr": float(line_re_module[0][6])*1000,
                    "leak_pwr": float(line_re_module[0][8])*1000,
                    "total_pwr": float(line_re_module[0][10])*1000,
                    "percent": float(line_re_module[0][12]),
                }

    ## rest part in addition to modules
    inter_m, switch_m, leak_m, total_m, percent_m = 0, 0, 0, 0, 0
    for module, pwr in pwr_dict.items():
        if module == design_top:
            inter_pwr_total = pwr["inter_pwr"]
            switch_pwr_total = pwr["switch_pwr"]
            leak_pwr_total = pwr["leak_pwr"]
            total_pwr_total = pwr["total_pwr"]
            percent_total = pwr["percent"]
        else:
            inter_m += pwr["inter_pwr"]
            switch_m += pwr["switch_pwr"]
            leak_m += pwr["leak_pwr"]
            total_m += pwr["total_pwr"]
            percent_m += pwr["percent"]
    inter_pwr_rest = inter_pwr_total - inter_m
    switch_pwr_rest = switch_pwr_total - switch_m
    leak_pwr_rest = leak_pwr_total - leak_m
    total_pwr_rest = total_pwr_total - total_m
    percent_rest = percent_total - percent_m

    inter_pwr_rest = 0 if inter_pwr_rest < 0.5 else inter_pwr_rest
    switch_pwr_rest = 0 if switch_pwr_rest < 0.5 else switch_pwr_rest
    leak_pwr_rest = 0 if leak_pwr_rest < 0.5 else leak_pwr_rest
    total_pwr_rest = 0 if total_pwr_rest < 0.5 else total_pwr_rest
    percent_rest = 0 if percent_rest < 0.1 else percent_rest

    pwr_dict[f"{design_top}_rest"] = {
        "inter_pwr": inter_pwr_rest,
        "switch_pwr": switch_pwr_rest,
        "leak_pwr": leak_pwr_rest,
        "total_pwr": total_pwr_rest,
        "percent": percent_rest,
    }
    print(f"Power Report {pwr_dict}")
    return pwr_dict


def parse_power_group(pwr_rpt_path):
    ## ================= Total Power Report (power group) =================
    pwr_dict = {}
    with open (pwr_rpt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_re_top = re.findall(f"(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)\((\s*)(\S+)%\)", line)
            if line_re_top:
                pwr_group = line_re_top[0][0]
                if not pwr_group in ['clock_network', 'register', 'combinational', 'sequential', 'memory', 'io_pad', 'black_box']:
                    continue
                # print(line_re_top)
                pwr_dict[pwr_group] = {
                    "inter_pwr": float(line_re_top[0][2])*1000,
                    "switch_pwr": float(line_re_top[0][4])*1000,
                    "leak_pwr": float(line_re_top[0][6])*1000,
                    "total_pwr": float(line_re_top[0][8])*1000,
                    "percent": float(line_re_top[0][11]),
                }
    
    print(f"Power Report for {pwr_rpt_path}: {pwr_dict}")
    return pwr_dict



def process_module(args):
    module_name, flow_dir, design_top, design_name = args
    results = {}
    
    # Process total power report
    rpt_path = f"./rpt_data/net/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/module/{module_name}.total_pwr.rpt"
    results['group'] = parse_power_group(rpt_path)
    
    return module_name, results


def update_module_pwr_rpt(pwr_module_rpt, design_top):
    pwr_module_rpt[f"{design_top}_rest"] = {
            "group": {}
    }
    for grp in ['clock_network', 'register', 'combinational', 'sequential', 'memory', 'io_pad', 'black_box']:
        inter_m, switch_m, leak_m, total_m = 0, 0, 0, 0
        for module, pwr in pwr_module_rpt.items():
            if module == design_top or module == f"{design_top}_rest":
                continue
           
            inter_m += pwr["group"][grp]["inter_pwr"]
            switch_m += pwr["group"][grp]["switch_pwr"]
            leak_m += pwr["group"][grp]["leak_pwr"]
            total_m += pwr["group"][grp]["total_pwr"]


        inter_pwr_rest = pwr_module_rpt[design_top]['group'][grp]['inter_pwr'] - inter_m
        switch_pwr_rest = pwr_module_rpt[design_top]['group'][grp]["switch_pwr"] - switch_m
        leak_pwr_rest = pwr_module_rpt[design_top]['group'][grp]["leak_pwr"] - leak_m
        total_pwr_rest = pwr_module_rpt[design_top]['group'][grp]["total_pwr"] - total_m
        inter_pwr_rest = 0 if inter_pwr_rest < 0.5 else inter_pwr_rest
        switch_pwr_rest = 0 if switch_pwr_rest < 0.5 else switch_pwr_rest
        leak_pwr_rest = 0 if leak_pwr_rest < 0.5 else leak_pwr_rest
        total_pwr_rest = 0 if total_pwr_rest < 0.5 else total_pwr_rest
        
        percent_rest = 0
        

        pwr_module_rpt[f"{design_top}_rest"]['group'][grp] = {
            "inter_pwr": inter_pwr_rest,
            "switch_pwr": switch_pwr_rest,
            "leak_pwr": leak_pwr_rest,
            "total_pwr": total_pwr_rest,
            "percent": percent_rest,
        }

    return pwr_module_rpt



def autoRun(design_name, design_top):
    print('Current Design:', design_name)
    print("Hierarchical Power Report")
    rpt_path = f"./rpt_data/net/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/{design_top}.report_power_hier.rpt"
    pwr_dict_hiearch = parse_hierarch_pwr(design_top, rpt_path)
    print(pwr_dict_hiearch.keys())


    print("Power Group Report")
    rpt_path = f"./rpt_data/net/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/{design_top}.report_power.rpt"
    pwr_dict_group = parse_power_group(rpt_path)


    print("Module Power Report")
    flow_dir = ""
    modules = [module for module in pwr_dict_hiearch.keys() if module != design_top+"_rest"]
    args = [(module, flow_dir, design_top, design_name) for module in modules]

    with Pool() as pool:
        pwr_module_rpt = dict(pool.map(process_module, args))

    pwr_module_rpt = update_module_pwr_rpt(pwr_module_rpt, design_top)

    global module_cnt
    module_cnt += len(modules)


    
    pwr_dct = {
        "hierarch" : pwr_dict_hiearch,
        "group" : pwr_dict_group,
        "module" : pwr_module_rpt,
    }
    if not os.path.exists(f"./rtl_pwr_data/net_pwr_rpt_{phase}"):
        os.makedirs(f"./rtl_pwr_data/net_pwr_rpt_{phase}")
    with open(f"./rtl_pwr_data/net_pwr_rpt_{phase}/{design_name}.json", 'w') as f:
        json.dump(pwr_dct, f, indent=4)

    
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

    global phase

    phase = 'route'

    for phase in ['init', 'route']:

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