import os, time, json, re, pickle
from multiprocessing import Pool
design_json = "/data/wenjifang/AIG_analyzer/LS-benchmark/design_rtl_timer.json"

def run_one_design(design_name):
    print('Current Design: ', design_name)
    ep_feat_dir = f"../data/feat/{design_name}.pkl"
    with open (ep_feat_dir, 'rb') as f:
        ep_feat_dict = pickle.load(f)
    
    v2n_map_dir = f"../../generated_name_map/{design_name}_name_map.json"
    with open (v2n_map_dir, 'r') as f:
        v2n_map_dict_re = json.load(f)
    v2n_map_dict = dict(zip(v2n_map_dict_re.values(), v2n_map_dict_re.keys()))

    v2n_map_dir = f"../../generated_name_map/{design_name}_name_map2.json"
    with open (v2n_map_dir, 'r') as f:
        v2n_map_dict2 = json.load(f)

    dc_label_dir = f"/data/wenjifang/vlg2netlist/dc_stat/data/EP_data/{tool}_ep_label/{design_name}_{phase}.pkl"
    with open (dc_label_dir, 'rb') as f:
        dc_label_dict = pickle.load(f)
    
    
    ep_mapping_dir = '/data/wenjifang/masterRTL2/DC_label/DC_mapping_dict'
    with open(f'{ep_mapping_dir}/{design_name}.json', 'r') as f:
        dc_mapping_dict = json.load(f)
    
    mapped_label_dict = {}
    for ep, vec in dc_label_dict.items():
        if ep in dc_mapping_dict:
            ep_mapped = dc_mapping_dict[ep]
            mapped_label_dict[ep_mapped] = vec

            



    i = 0
    feat_lst, label_lst = [], []
    feat_dict, label_dict = {}, {}

    ep_feat_dict_final = {}
    for ep, vec in ep_feat_dict.items():
        ep = v2n_map_dict[ep]
        if ep in v2n_map_dict2.keys():
            ep = v2n_map_dict2[ep]
        ep_feat_dict_final[ep] = vec
    
    for ep_label, label_vec in mapped_label_dict.items():
        if ep_label in ep_feat_dict_final:
            vec = ep_feat_dict_final[ep_label]
            i += 1
            label = label_vec[bit]
            feat_lst.append(vec)
            label_lst.append(label)
            feat_dict[ep_label] = vec
            label_dict[ep_label] = label
    
    


    print(round(i/len(mapped_label_dict)*100))

    with open (f'./feat_label_data/pair/{design_name}_{phase}_feat.pkl', 'wb') as f:
        pickle.dump(feat_lst, f)
    with open (f'./feat_label_data/pair/{design_name}_{phase}_label.pkl', 'wb') as f:
        pickle.dump(label_lst, f)

    with open (f'./feat_label_data/pair/{design_name}_{phase}_feat_dict.pkl', 'wb') as f:
        pickle.dump(feat_dict, f)
    with open (f'./feat_label_data/pair/{design_name}_{phase}_label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)

    feat_all.extend(feat_lst)
    label_all.extend(label_lst)

def run_all(bench, design_name=None):
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        if design_name:
            if k == design_name:
                run_one_design(k)
        else:
            run_one_design(k)

def run_all_parallel(bench):
      with open(design_json, 'r') as f:
            design_data = json.load(f)
            bench_data = design_data[bench]
      with Pool(20) as p:
            p.map(run_one_design, list(bench_data.keys()))
            p.close()
            p.join()


if __name__ == '__main__':

    global phase, bit, tool
    phase = 'PREOPT'
    phase = 'SYN'

    if phase == 'PREOPT':
        tool = 'innovus'
    else:
        tool = 'dc'
    tpe_lst = ["delay", "len", "fanout", "cap", "tran", "rank"]
    tpe_lst = ["delay"]
    for tpe in tpe_lst:
        if tpe == "delay":
            bit = 0
        elif tpe == "len":
            bit = 1
        elif tpe == "fanout":
            bit = 2
        elif tpe == "cap":
            bit = 3
        elif tpe == "tran":
            bit = 4
        elif tpe == "rank":
            bit = 12

        print('Phase: ', phase)
        print('Type: ', tpe)

        bench_list_all = ['itc','opencores','VexRiscv','chipyard', 'riscvcores','NVDLA']

        design_name = 'TinyRocket'
        design_name = ""

        global feat_all, label_all
        feat_all, label_all = [], []
        
        for bench in bench_list_all:
            run_all(bench, design_name)
            # run_all_parallel(bench)
        
        # with open (f'./feat_label_data/all/feat_all_{phase}_{tpe}.pkl', 'wb') as f:
        #     pickle.dump(feat_all, f)
        # with open (f'./feat_label_data/all/label_all_{phase}_{tpe}.pkl', 'wb') as f:
        #     pickle.dump(label_all, f)