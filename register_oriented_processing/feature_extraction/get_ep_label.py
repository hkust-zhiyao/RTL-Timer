import json, os, re, copy, pickle
from timing_path import timing_path
import difflib

def get_rank(idx, ll):
    l_0 = ll*0
    l_5 = ll*0.05
    l_10 = ll*0.1
    l_20 = ll*0.2
    l_40 = ll*0.4
    l_60 = ll*0.6
    l_80 = ll*0.8
    l_100 = ll

    if l_0 <= idx <= l_5:
        ret_rank = 0
    elif l_5 < idx <= l_10:
        ret_rank = 1
    elif l_10 < idx <= l_20:
        ret_rank = 2
    elif l_20 < idx <= l_40:
        ret_rank = 3
    elif l_40 < idx <= l_60:
        ret_rank = 4
    elif l_60 < idx <= l_80:
        ret_rank = 5
    elif l_80 < idx <= l_100:
        ret_rank = 6
    return ret_rank

def get_dc_label(bench, design_name, design_top, clk_name):
    print('Current Design: ', design_name)
    feat_dict = {}
    ret_ep_lst = []
    
    rpt_dir = f"/home/coguest5/RTL-Timer/dataset/netlist/netlist_rpt"
    if phase == 'SYN':
        sta_rpt_dir = f"{rpt_dir}/{design_name}.timing.rpt"
    # else:
    #     sta_rpt_dir = f"{rpt_dir}/{design_top}_{design_name}_{phase}_TYP_SAIF_SDF/{design_top}.timing_gba.rpt"
    with open (sta_rpt_dir, 'r') as f:
        lines = f.readlines()
    
    lines_new = []
    for line in lines:
        line = re.sub(r"\&", "", line)
        lines_new.append(line)
    lines = lines_new
    
    path_cnt = 0
    for idx, line in enumerate(lines):
        s = re.findall(r"^  Startpoint: (\S+)", line)
        e = re.findall(r"^  Endpoint: (\S+)", line)
        s_line = re.findall(r"^  Point(\s+)Fanout(\s+)Cap(\s+)Trans(\s+)Incr(\s+)Path(.*)", line)
        e_line = re.findall(r"^  data arrival time(.*)", line)
        if s:
            start = s[0]
        elif e:
            ep = e[0]
        elif s_line:
            ret_ep_lst.append(copy.copy(ep))
            # ep_re = re.findall(r'_reg_(\d+)_', ep)
            # if ep_re:
            #     bit_num = ep_re[0]
            #     ep = re.sub(r'_reg_(\d+)_', f'_reg[{bit_num}]', ep)
            path = timing_path(start, ep)
            node_name = None
            while not e_line:
                e_line = re.findall(r"^  data arrival time(.*)", lines[idx])
                node_name = path.add_cell(lines[idx], node_name)
                idx += 1
            path_cnt += 1
            feat_vec = path.get_path_feat()
            feat_dict[ep] = feat_vec
    
    ep_lst = list(feat_dict.keys())
    ep_len = len(ep_lst)

    feat_dict_final = {}
    for idx, ep in enumerate(ep_lst):
        
        vec = feat_dict[ep]
        vec.append(ep_len)
        rank_num = get_rank(idx, ep_len)
        vec.append(rank_num)
        if (ep_len == None) or (rank_num == None):
            print(ep_len, rank_num)
            assert False
        feat_dict_final[ep] = vec

        # print(vec)

    with open (f'/home/coguest5/RTL-Timer/dataset/netlist/label/{design_name}_{phase}.pkl', 'wb') as f:
        pickle.dump(feat_dict_final, f)
    
    print(len(ret_ep_lst))
    
    return feat_dict_final


def run_one_design(bench, design_name, design_top, clk_name):
    label_dict = get_dc_label(bench, design_name, design_top, clk_name)

    with open (f"/home/coguest5/RTL-Timer/dataset/BOG/SOG/feat/{design_name}.pkl", "rb") as f:
        sog_ep_dct = pickle.load(f)
    
    idx = 0

    ep_set = set()

    for ep, val in sog_ep_dct.copy().items():
        ep_new = re.sub(r'\$_(\w)*DFF(\w)*_(\S+)$', "", ep)
        if ep_new != ep:
            sog_ep_dct[ep_new] = val
            del sog_ep_dct[ep]
    
    for ep, val in label_dict.copy().items():
        ep_new = re.sub(f"/", "_", ep)
        if ep_new != ep:
            label_dict[ep_new] = val
            del label_dict[ep]
    
    for ep in sog_ep_dct.keys():
        if ep in label_dict.keys():
            idx += 1
        # else:
        #     print(ep)
    
    # print(len(sog_ep_dct), len(label_dict))
    ll_label = len(label_dict)

    final_feat_dct, final_label_dct = {}, {}

    for ep in label_dict.copy().keys():
        if ep in sog_ep_dct.copy().keys():
            final_feat_dct[ep] = sog_ep_dct[ep]
            final_label_dct[ep] = label_dict[ep]
            del sog_ep_dct[ep]
            del label_dict[ep]
    
    for ep in label_dict.keys():
        ep_match_lst = difflib.get_close_matches(ep, list(sog_ep_dct.keys()), n=1, cutoff=0.9)
        # print(ep, ep_match_lst)
        if not ep_match_lst:
            continue
        ep_match = ep_match_lst[0]
        final_feat_dct[ep_match] = sog_ep_dct[ep_match]
        final_label_dct[ep_match] = label_dict[ep]

    # print(len(final_feat_dct), len(final_label_dct))

    print(f'Register mapping coverage: {round(len(final_label_dct)/ll_label,2)*100}%')

    with open (f"/home/coguest5/RTL-Timer/modeling/feat_label/bit-wise/{design_name}.pkl", "wb") as f:
        pickle.dump((final_feat_dct, final_label_dct), f)


def run_one_bench(bench, design_data, name=None):
    bench_root = design_data[bench]
    for k, v in bench_root.items():
        if name:
            if k == name:
                design_top = v[0]
                clk_name = v[1]
                run_one_design(bench, k, design_top, clk_name)
        else:
            design_top = v[0]
            clk_name = v[1]
            run_one_design(bench, k, design_top, clk_name)


if __name__ == '__main__':
    design_json = f"/home/coguest5/LS-benchmark/design_rtl_timer.json"
    global phase, tool
    phase = 'SYN'
    # phase = 'PREOPT'

    if phase == 'SYN':
        tool = 'dc'
    elif phase == 'PREOPT':
        tool = 'innovus'

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    design_name = "b17"
    design_name = ""
    bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']

    # for bench in bench_list:
    #     run_one_bench(bench, design_data, design_name)

    design_lst = []
    for bench in bench_list:
        bench_root = design_data[bench]
        for k, v in bench_root.items():
            design_lst.append(k)
        
    print(design_lst)