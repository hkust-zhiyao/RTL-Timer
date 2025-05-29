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
    # percent_rest = 0 if percent_rest < 0.1 else percent_rest

    pwr_dict[f"{design_top}_rest"] = {
        "inter_pwr": inter_pwr_rest,
        "switch_pwr": switch_pwr_rest,
        "leak_pwr": leak_pwr_rest,
        "total_pwr": total_pwr_rest,
        "percent": percent_rest,
    }
    # print(f"Power Report {pwr_dict}")
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
    
    # print(f"Power Report: {pwr_dict}")
    return pwr_dict


def parse_cell_pwr(rpt_path):
    ## ================= Cell Power Report =================
    pwr_dict = {}
    with open (rpt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_re_cell = re.findall(r"(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)\((\s*)(\S+)%\)(\s+)(\S+)", line)
            if line_re_cell:
                # input()
                cell_name = line_re_cell[0][0]
                pwr_dict[cell_name] = {
                    "cell_type": line_re_cell[0][2],
                    "inter_pwr": float(line_re_cell[0][4])*1000,
                    "switch_pwr": float(line_re_cell[0][6])*1000,
                    "leak_pwr": float(line_re_cell[0][8])*1000,
                    "total_pwr": float(line_re_cell[0][10])*1000,
                    "percent": float(line_re_cell[0][13]),
                    "area": float(line_re_cell[0][15]),
                }
            line_re_total = re.findall(r"Totals \((\S+) cells\)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)\((\s*)(\S+)%\)(\s+)(\S+)", line)
            if line_re_total:
                pwr_dict["TOTAL"] = {
                    "cell_cnt": float(line_re_total[0][0])*1000,
                    "inter_pwr": float(line_re_total[0][2])*1000,
                    "switch_pwr": float(line_re_total[0][4])*1000,
                    "leak_pwr": float(line_re_total[0][6])*1000,
                    "total_pwr": float(line_re_total[0][8])*1000,
                    "percent": float(line_re_total[0][11]),
                    "area": float(line_re_total[0][13]),
                }



    # === register feature vector ====
    ## 0. #.DFF 1. max dyn_pwr 2. min dyn_pwr 3. avg dyn_pwr
    ## 4. max stat_pwr 5. min stat_pwr 6. avg stat_pwr
    ## 7. max total_pwr 8. min total_pwr 9. avg total_pwr
    ## 10. max area 11. min area 12. avg area

    # === combinational feature vector ====
    ## 0. #.And 1. #.Or 2. #.Inv 3. #.Xor 4. #.Mux 5. total logic cell number
    ## 6. max dyn_pwr 7. min dyn_pwr 8. avg dyn_pwr
    ## 9. max stat_pwr 10. min stat_pwr 11. avg stat_pwr
    ## 12. max total_pwr 13. min total_pwr 14. avg total_pwr
    ## 15. max area 16. min area 17. avg area

    # === total feature vector ====
    ## 0. #.DFF 1. #.And 2. #.Or 3. #.Inv 4. #.Xor 5. #.Mux 6. total cell number
    ## 7. max dyn_pwr 8. min dyn_pwr 9. avg dyn_pwr
    ## 10. max stat_pwr 11. min stat_pwr 12. avg stat_pwr
    ## 13. max total_pwr 14. min total_pwr 15. avg total_pwr
    ## 16. max area 17. min area 18. avg area
    feat_vec_reg = [0] * 13
    feat_vec_comb = [0] * 18
    feat_vec_total = [0] * 19
    reg_dyn_pwr_list, reg_stat_pwr_list, reg_total_pwr_list, reg_area_list = [], [], [], []
    comb_dyn_pwr_list, comb_stat_pwr_list, comb_total_pwr_list, comb_area_list = [], [], [], []
    total_dyn_pwr_list, total_stat_pwr_list, total_total_pwr_list, total_area_list = [], [], [], []
    for cell in pwr_dict.keys():
        if cell == "TOTAL":
                continue
        cell_type = pwr_dict[cell]["cell_type"]
        if cell_type == "DFF_X1" or cell_type == "DFFSR_X1" or cell_type == "DFFRS_X1":
            feat_vec_reg[0] += 1
            feat_vec_total[0] += 1
            reg_dyn_pwr_list.append(pwr_dict[cell]["inter_pwr"] + pwr_dict[cell]["switch_pwr"])
            reg_stat_pwr_list.append(pwr_dict[cell]["leak_pwr"])
            reg_total_pwr_list.append(pwr_dict[cell]["total_pwr"])
            reg_area_list.append(pwr_dict[cell]["area"])
            
        else:
            if cell_type == "AND2_X1":
                feat_vec_comb[0] += 1
                feat_vec_total[1] += 1
            elif cell_type == "OR2_X1":
                feat_vec_comb[1] += 1
                feat_vec_total[2] += 1
            elif cell_type == "INV_X1":
                feat_vec_comb[2] += 1
                feat_vec_total[3] += 1
            elif cell_type == "XOR2_X1":
                feat_vec_comb[3] += 1
                feat_vec_total[4] += 1
            elif cell_type == "MUX2_X1":
                feat_vec_comb[4] += 1
                feat_vec_total[5] += 1
            comb_dyn_pwr_list.append(pwr_dict[cell]["inter_pwr"] + pwr_dict[cell]["switch_pwr"])
            comb_stat_pwr_list.append(pwr_dict[cell]["leak_pwr"])
            comb_total_pwr_list.append(pwr_dict[cell]["total_pwr"])
            comb_area_list.append(pwr_dict[cell]["area"])
        total_dyn_pwr_list.append(pwr_dict[cell]["inter_pwr"] + pwr_dict[cell]["switch_pwr"])
        total_stat_pwr_list.append(pwr_dict[cell]["leak_pwr"])
        total_total_pwr_list.append(pwr_dict[cell]["total_pwr"])
        total_area_list.append(pwr_dict[cell]["area"])

    
    
    if len(reg_dyn_pwr_list) > 0:  # Add safety check for empty lists
        feat_vec_reg[1] = max(reg_dyn_pwr_list)
        feat_vec_reg[2] = min(reg_dyn_pwr_list)
        feat_vec_reg[3] = sum(reg_dyn_pwr_list) / len(reg_dyn_pwr_list)
        feat_vec_reg[4] = max(reg_stat_pwr_list)
        feat_vec_reg[5] = min(reg_stat_pwr_list)
        feat_vec_reg[6] = sum(reg_stat_pwr_list) / len(reg_stat_pwr_list)
        feat_vec_reg[7] = max(reg_total_pwr_list)
        feat_vec_reg[8] = min(reg_total_pwr_list)
        feat_vec_reg[9] = sum(reg_total_pwr_list) / len(reg_total_pwr_list)
        feat_vec_reg[10] = max(reg_area_list)
        feat_vec_reg[11] = min(reg_area_list)
        feat_vec_reg[12] = sum(reg_area_list) / len(reg_area_list)

    if len(comb_dyn_pwr_list) > 0:  # Add safety check for empty lists
        feat_vec_comb[5] = len(pwr_dict) - 1
        feat_vec_comb[6] = max(comb_dyn_pwr_list)
        feat_vec_comb[7] = min(comb_dyn_pwr_list)
        feat_vec_comb[8] = sum(comb_dyn_pwr_list) / len(comb_dyn_pwr_list)
        feat_vec_comb[9] = max(comb_stat_pwr_list)
        feat_vec_comb[10] = min(comb_stat_pwr_list)
        feat_vec_comb[11] = sum(comb_stat_pwr_list) / len(comb_stat_pwr_list)
        feat_vec_comb[12] = max(comb_total_pwr_list)
        feat_vec_comb[13] = min(comb_total_pwr_list)
        feat_vec_comb[14] = sum(comb_total_pwr_list) / len(comb_total_pwr_list)
        feat_vec_comb[15] = max(comb_area_list)
        feat_vec_comb[16] = min(comb_area_list)
        feat_vec_comb[17] = sum(comb_area_list) / len(comb_area_list)

    if len(total_dyn_pwr_list) > 0:  # Add safety check for empty lists
        feat_vec_total[6] = len(pwr_dict) - 1
        feat_vec_total[7] = max(total_dyn_pwr_list)
        feat_vec_total[8] = min(total_dyn_pwr_list)
        feat_vec_total[9] = sum(total_dyn_pwr_list) / len(total_dyn_pwr_list)
        feat_vec_total[10] = max(total_stat_pwr_list)
        feat_vec_total[11] = min(total_stat_pwr_list)   
        feat_vec_total[12] = sum(total_stat_pwr_list) / len(total_stat_pwr_list)
        feat_vec_total[13] = max(total_total_pwr_list)
        feat_vec_total[14] = min(total_total_pwr_list)
        feat_vec_total[15] = sum(total_total_pwr_list) / len(total_total_pwr_list)
        feat_vec_total[16] = max(total_area_list)
        feat_vec_total[17] = min(total_area_list)
        feat_vec_total[18] = sum(total_area_list) / len(total_area_list)


    

    return feat_vec_reg, feat_vec_comb, feat_vec_total


def parse_net_pwr(rpt_path):
    ## ================= Cell Power Report =================
    pwr_dict = {}
    with open (rpt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_re_net = re.findall(r"(\S+)(\s+)(1.10)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)", line)
            if line_re_net:
                net_name = line_re_net[0][0]
                pwr_dict[net_name] = {
                    "net_load": float(line_re_net[0][4]),
                    "stat_prob": float(line_re_net[0][6]),
                    "toggle_rate": float(line_re_net[0][8]),
                    "switch_pwr": float(line_re_net[0][10])*1000
                }

            line_re_total = re.findall(r"Total \((\S+) nets\)(\s+)(\S+) Watt", line)
            if line_re_total:
                pwr_dict["TOTAL"] = {
                    "net_cnt": float(line_re_total[0][0]),
                    "switch_pwr": float(line_re_total[0][2])*1000,
                }
    # print(f"Power Report for {rpt_path}: {pwr_dict}")
    # print(pwr_dict['TOTAL'])


    ## get net report feature vector
    feat_vec = [0] * 13
    feat_vec_reg = [0] * 13
    feat_vec_comb = [0] * 13
    ## 0. #.net 
    # 1. max net_load 2. min net_load 3. avg net_load
    # 4. max stat_prob 5. min stat_prob 6. avg stat_prob
    # 7. max toggle_rate 8. min toggle_rate 9. avg toggle_rate
    # 10. max switch_pwr 11. min switch_pwr 12. avg switch_pwr
    net_load_list, stat_prob_list, toggle_rate_list, switch_pwr_list = [], [], [], []
    reg_net_load_list, reg_stat_prob_list, reg_toggle_rate_list, reg_switch_pwr_list = [], [], [], []
    comb_net_load_list, comb_stat_prob_list, comb_toggle_rate_list, comb_switch_pwr_list = [], [], [], []
    for net in pwr_dict.keys():
        if net == "TOTAL":
            continue
        net_load_list.append(pwr_dict[net]["net_load"])
        stat_prob_list.append(pwr_dict[net]["stat_prob"])
        toggle_rate_list.append(pwr_dict[net]["toggle_rate"])
        switch_pwr_list.append(pwr_dict[net]["switch_pwr"])
        if re.findall(r"^_(\S+)_$", net):
            feat_vec_comb[0] += 1
            comb_net_load_list.append(pwr_dict[net]["net_load"])
            comb_stat_prob_list.append(pwr_dict[net]["stat_prob"])
            comb_toggle_rate_list.append(pwr_dict[net]["toggle_rate"])
            comb_switch_pwr_list.append(pwr_dict[net]["switch_pwr"])
        else:
            feat_vec_reg[0] += 1
            reg_net_load_list.append(pwr_dict[net]["net_load"])
            reg_stat_prob_list.append(pwr_dict[net]["stat_prob"])
            reg_toggle_rate_list.append(pwr_dict[net]["toggle_rate"])
            reg_switch_pwr_list.append(pwr_dict[net]["switch_pwr"])
    if len(net_load_list) > 0:  # Add safety check for empty lists
        feat_vec[0] = len(pwr_dict) - 1
        feat_vec[1] = max(net_load_list)
        feat_vec[2] = min(net_load_list)
        feat_vec[3] = sum(net_load_list) / len(net_load_list)
        feat_vec[4] = max(stat_prob_list)
        feat_vec[5] = min(stat_prob_list)
        feat_vec[6] = sum(stat_prob_list) / len(stat_prob_list)
        feat_vec[7] = max(toggle_rate_list)
        feat_vec[8] = min(toggle_rate_list)
        feat_vec[9] = sum(toggle_rate_list) / len(toggle_rate_list)
        feat_vec[10] = max(switch_pwr_list)
        feat_vec[11] = min(switch_pwr_list)
        feat_vec[12] = sum(switch_pwr_list) / len(switch_pwr_list)

    if len(reg_net_load_list) > 0:  # Add safety check for empty lists
        feat_vec_reg[1] = max(reg_net_load_list)
        feat_vec_reg[2] = min(reg_net_load_list)
        feat_vec_reg[3] = sum(reg_net_load_list) / len(reg_net_load_list)
        feat_vec_reg[4] = max(reg_stat_prob_list)
        feat_vec_reg[5] = min(reg_stat_prob_list)
        feat_vec_reg[6] = sum(reg_stat_prob_list) / len(reg_stat_prob_list)
        feat_vec_reg[7] = max(reg_toggle_rate_list)
        feat_vec_reg[8] = min(reg_toggle_rate_list)
        feat_vec_reg[9] = sum(reg_toggle_rate_list) / len(reg_toggle_rate_list)
        feat_vec_reg[10] = max(reg_switch_pwr_list)
        feat_vec_reg[11] = min(reg_switch_pwr_list)
        feat_vec_reg[12] = sum(reg_switch_pwr_list) / len(reg_switch_pwr_list)
    
    if len(comb_net_load_list) > 0:
        feat_vec_comb[1] = max(comb_net_load_list)
        feat_vec_comb[2] = min(comb_net_load_list)
        feat_vec_comb[3] = sum(comb_net_load_list) / len(comb_net_load_list)
        feat_vec_comb[4] = max(comb_stat_prob_list)
        feat_vec_comb[5] = min(comb_stat_prob_list)
        feat_vec_comb[6] = sum(comb_stat_prob_list) / len(comb_stat_prob_list)
        feat_vec_comb[7] = max(comb_toggle_rate_list)
        feat_vec_comb[8] = min(comb_toggle_rate_list)
        feat_vec_comb[9] = sum(comb_toggle_rate_list) / len(comb_toggle_rate_list)
        feat_vec_comb[10] = max(comb_switch_pwr_list)
        feat_vec_comb[11] = min(comb_switch_pwr_list)
        feat_vec_comb[12] = sum(comb_switch_pwr_list) / len(comb_switch_pwr_list)


    return feat_vec, feat_vec_reg, feat_vec_comb

def process_module(args):
    module_name, flow_dir, cmd, design_top, design_name = args
    results = {}
    
    # Process total power report
    rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/module/{module_name}.total_pwr.rpt"
    results['group'] = parse_power_group(rpt_path)
    
    # Process cell power report
    rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/module/{module_name}.cell_pwr.rpt"
    feat_reg, feat_comb, feat_total = parse_cell_pwr(rpt_path)
    results['cell'] = feat_total
    results['cell_reg'] = feat_reg
    results['cell_comb'] = feat_comb
    
    # Process net power report
    rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/module/{module_name}.net_pwr.rpt"
    feat_reg, feat_comb, feat_total = parse_net_pwr(rpt_path)
    results['net'] = feat_total
    results['net_reg'] = feat_reg
    results['net_comb'] = feat_comb
    
    return module_name, results


def update_module_pwr_rpt(pwr_module_rpt, design_top):
    pwr_module_rpt[f"{design_top}_rest"] = {
            "group": {},
            "cell": pwr_module_rpt[design_top]['cell'].copy(),
            "cell_reg": pwr_module_rpt[design_top]['cell_reg'].copy(),
            "cell_comb": pwr_module_rpt[design_top]['cell_comb'].copy(),
            "net": pwr_module_rpt[design_top]['net'].copy(),
            "net_reg": pwr_module_rpt[design_top]['net_reg'].copy(),
            "net_comb": pwr_module_rpt[design_top]['net_comb'].copy(),
    }

    ### Note: cell count of top module does not include the submodules, need to aggregate for total count!

    ### update cell/net count
    cell_vec_all = pwr_module_rpt[design_top]['cell']
    cell_vec_all_reg = pwr_module_rpt[design_top]['cell_reg']
    cell_vec_all_comb = pwr_module_rpt[design_top]['cell_comb']
    net_vec_all = pwr_module_rpt[design_top]['net']
    net_vec_all_reg = pwr_module_rpt[design_top]['net_reg']
    net_vec_all_comb = pwr_module_rpt[design_top]['net_comb']
    for module, pwr in pwr_module_rpt.items():
        if module == design_top or module == f"{design_top}_rest":
            continue
        cell_vec_m = pwr['cell']
        cell_vec_reg_m = pwr['cell_reg']
        cell_vec_comb_m = pwr['cell_comb']
        net_vec_m = pwr['net']
        net_vec_reg_m = pwr['net_reg']
        net_vec_comb_m = pwr['net_comb']
        
        ## only subtract the cell count (0-6)
        for i in range(7):
            cell_vec_all[i] += cell_vec_m[i]
        for i in range(1):
            cell_vec_all_reg[i] += cell_vec_reg_m[i]
        for i in range(6):
            cell_vec_all_comb[i] += cell_vec_comb_m[i]
        ## subtract the net count (0-12)
        for i in range(1):
            net_vec_all[i] += net_vec_m[i]
            net_vec_all_comb[i] += net_vec_comb_m[i]
            net_vec_all_reg[i] += net_vec_reg_m[i]
    pwr_module_rpt[design_top]['cell'] = cell_vec_all
    pwr_module_rpt[design_top]['cell_reg'] = cell_vec_all_reg
    pwr_module_rpt[design_top]['cell_comb'] = cell_vec_all_comb
    pwr_module_rpt[design_top]['net'] = net_vec_all
    pwr_module_rpt[design_top]['net_reg'] = net_vec_all_reg
    pwr_module_rpt[design_top]['net_comb'] = net_vec_all_comb



    ### update power group
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
    rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/{design_top}.report_power_hier.rpt"
    pwr_dict_hiearch = parse_hierarch_pwr(design_top, rpt_path)
    # print(pwr_dict_hiearch.keys())


    print("Power Group Report")
    rpt_path = f"./rpt_data/BOG/{cmd}/{design_top}_{design_name}_TYP_SYN_TYP_SAIF_SDF/{design_top}.report_power.rpt"
    pwr_dict_group = parse_power_group(rpt_path)


    print("Module Power Report")
    flow_dir = ""
    modules = [module for module in pwr_dict_hiearch.keys() if module != design_top+"_rest"]
    args = [(module, flow_dir, cmd, design_top, design_name) for module in modules]

    # with Pool() as pool:
    #     pwr_module_rpt = dict(pool.map(process_module, args))
    pwr_module_rpt = {}
    for module in modules:
        module_name, results = process_module((module, flow_dir, cmd, design_top, design_name))
        pwr_module_rpt[module_name] = results

    ## get module rest part
    pwr_module_rpt = update_module_pwr_rpt(pwr_module_rpt, design_top)
    

    global module_cnt
    module_cnt += len(modules)

    pwr_dct = {
        "hierarch" : pwr_dict_hiearch,
        "group" : pwr_dict_group,
        "module" : pwr_module_rpt,
    }

    if not os.path.exists(f"./rtl_pwr_data/bog_pwr_rpt/{cmd}"):
        os.makedirs(f"./rtl_pwr_data/bog_pwr_rpt/{cmd}")
    with open(f"./rtl_pwr_data/bog_pwr_rpt/{cmd}/{design_name}_{cmd}.json", 'w') as f:
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

    global phase, cmd
    cmd = 'sog'
    cmd = 'aig'
    cmd = 'aimg'
    cmd = 'xag'

    for cmd in ['sog', 'aig', 'aimg', 'xag']:


        phase = 'SYN'
        assert phase in ['SYN', 'PREOPT', 'PLACE', 'CTS', 'ROUTE']

        with open(design_json, 'r') as f:
            design_data = json.load(f)

        
        design_name = "TinyRocket"
        bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
        # bench_list = ['riscvcores', 'NVDLA']

        global module_cnt
        module_cnt = 0
        
        for bench in bench_list:
            run_one_bench(bench, design_data, design_name)


        print(f"Total Module Count: {module_cnt}")