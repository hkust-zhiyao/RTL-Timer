import os, re, json
from multiprocessing import Pool
from clean_replace_vlg import clean_vlg, clean_map_vlg
design_json = "/home/coguest5/LS-benchmark/design_rtl_timer.json"

def run_one_design(para):
    bench, design, top_name = para
    print(f"Running {design}")
    ys_template = f"./run_ys_template.ys"
    ys_scr = f"./run_{design}.ys"
    with open(ys_template, "r") as f:
        lines = f.readlines()
    
    ### change path here ###
    ## 1. original RTL code path
    ## change the RTL path first
    design_path = f"/home/coguest5/LS-benchmark/{bench}/rtl/{design}/"
    ## 2. BOG output path
    save_path = f"/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/generated_netlist_file/{top_name}_{design}_TYP.syn.v"
    mapped_save_path = f"/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/mapped_netlist/{top_name}_{design}_TYP.syn.v"
    #######################
    if bench in ['itc']:
        read_line = f"read  -verific;\nread -vhdl {design_path}/{design}.vhd\n"
    else:
        read_line = f"read -verific;\nread_verilog {design_path}/*.v\n"

    with open(ys_scr, "w") as f_scr:
        f_scr.writelines(read_line)
        for line in lines:
            line = line.replace("design_top", top_name)
            line = line.replace("lib_name", f"nangate45_{cmd}.lib")
            line = line.replace("save_path", save_path)
            f_scr.writelines(line)
    os.system(f"yosys {ys_scr}")
    os.system(f"rm -rf {ys_scr}")

    os.system(f'cp ./sdc_template.sdc /home/coguest5/RTL-Timer/dataset/BOG/{cmd}/generated_sdc_file/{top_name}_{design}_TYP.sdc')

    clean_map_vlg(save_path, mapped_save_path)



def run_all(bench, design_name=None):
    
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for name, v in bench_data.items():
        top_name = v[0]
        clk = v[1]
        reset = v[2]
        para = (bench, name, top_name)
        if design_name:
            if name == design_name:
                run_one_design(para)
        else:
            run_one_design(para)




def run_all_parallel(bench):
    para_lst = []
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for name, v in bench_data.items():
        top_name = v[0]
        para = (bench, name, top_name)
        para_lst.append(para)
    
    with Pool(20) as p:
        p.map(run_one_design, para_lst)
        p.close()
        p.join()



if __name__ == '__main__':
    
    global cmd
    cmd = 'SOG'
    assert cmd in ['SOG', 'AIG', 'AIMG', 'XAG']


    bench_list_all = ['itc','opencores','VexRiscv', 'chipyard', 'riscvcores','NVDLA']
    design_name = ''

    
    for bench in bench_list_all:
        # run_all(bench, design_name)
        run_all_parallel(bench)
