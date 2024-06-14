import re, os, time, json, pickle
from timing_path import timing_path


bench_dir = "/home/coguest5/LS-benchmark"

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


def get_rank_percent(delay, delay_max):
    delay_max = max(delay, delay_max)
    pc = (delay_max - delay)/delay_max
    return pc, delay_max
    
def autoRun(bench, design_name, design_top, clk_name, user=None):
    print('Current Design: ', design_name)
    feat_dict = {}

    flow_dir = f"/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/timing_rpt"
    sta_rpt_dir = f"{flow_dir}/{design_name}.timing.rpt"

    with open (sta_rpt_dir, 'r') as f:
        lines = f.readlines()
    
    path_cnt = 0
    for idx, line in enumerate(lines):
        s = re.findall(r"^  Startpoint: (\S+)", line)
        e = re.findall(r"^  Endpoint: (\S+)", line)
        s_line = re.findall(r"^  Point(\s+)Fanout(\s+)Cap(\s+)Trans(\s+)Incr(\s+)Path(.*)", line)
        e_line = re.findall(r"^  data arrival time(.*)", line)
        if s:
            start = s[0]
        elif e:
            end = e[0]
        elif s_line:
            path = timing_path(start, end)
            node_name = None
            while not e_line:
                e_line = re.findall(r"^  data arrival time(.*)", lines[idx])
                node_name = path.add_cell(lines[idx], node_name)
                idx += 1
            path_cnt += 1
            feat_vec = path.get_path_feat()
            feat_dict[end] = feat_vec

    
    ep_lst = list(feat_dict.keys())
    ep_len = len(ep_lst)

    feat_dict_final = {}
    delay_max = feat_dict[ep_lst[0]][0]
    for idx, ep in enumerate(ep_lst):
        vec = feat_dict[ep]
        vec.append(ep_len)
        rank_num = get_rank(idx, ep_len)
        rank_percent, delay_max = get_rank_percent(vec[0], delay_max)
        vec.append(rank_num)
        vec.append(rank_percent)
        feat_dict_final[ep] = vec
        
    
    # print(feat_dict_final)
    # exit()

    with open (f'/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/feat/{design_name}.pkl', 'wb') as f:
        pickle.dump(feat_dict_final, f)

    
def run_one_bench(bench, design_data, name=None):

    bench_root = design_data[bench]
    for k, v in bench_root.items():
        if name:
            if k == name:
                design_top = v[0]
                clk_name = v[1]
                autoRun(bench, k, design_top, clk_name)
        else:
            design_top = v[0]
            clk_name = v[1]
            autoRun(bench, k, design_top, clk_name)


if __name__ == '__main__':
    design_json = f"{bench_dir}/design_rtl_timer.json"

    global phase, cmd
    cmd = 'SOG'

    with open(design_json, 'r') as f:
        design_data = json.load(f)

    for phase in ['SYN']:
        design_name = "TinyRocket"
        design_name = ""
        bench_list = ['iscas', 'itc', 'opencores','VexRiscv', 'chipyard', 'riscvcores', 'NVDLA']
        for bench in bench_list:
            run_one_bench(bench, design_data, design_name)





    
