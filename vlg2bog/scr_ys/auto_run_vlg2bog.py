import os, re, json
from multiprocessing import Pool
from clean_replace_vlg import clean_replace_vlg

def run_one_design(param_lst):
    bench, design, top = param_lst
    print(f"Running {design}")
    ys_template = f"./run_ys_template.ys"
    ys_scr = f"./run_{design}.{cmd}.ys"
    with open(ys_template, "r") as f:
        lines = f.readlines()
        
    with open(ys_scr, "w") as f_scr:
        f_scr.writelines("read -verific\n")

        design_rtl_dir = f"../rtl_example/{bench}/rtl/{design}/"
        ## read every file in the design rtl dir
        for a, b, file_list in os.walk(design_rtl_dir):
            for file_name in file_list:
                if ('.v' == file_name[-2:]):
                    f_scr.writelines(f"read_verilog ../rtl_example/{bench}/rtl/{design}/{file_name}\n")
                elif ('.sv' == file_name[-3:]):
                    f_scr.writelines(f"read -sv ../rtl_example/{bench}/rtl/{design}/{file_name}\n")
                elif ('.vhd' == file_name[-4:]):
                    f_scr.writelines(f"read -vhdl ../rtl_example/{bench}/rtl/{design}/{file_name}\n")
        for line in lines:
            line = line.replace("top_name", top)
            line = line.replace("bench_name", bench)
            line = line.replace("design_name", design)
            line = line.replace("cmd", cmd)
            f_scr.writelines(line)
    os.system(f"yosys {ys_scr}")
    os.system(f"rm -rf {ys_scr}")

    bog_tmp_path = f"../bog_tmp/{design}.{cmd}.v"
    if not os.path.exists(f"../bog/{cmd}/"):
        os.makedirs(f"../bog/{cmd}/")
    bog_path = f"../bog/{cmd}/{design}.{cmd}.v"
    # os.system(f"cp {bog_tmp_path} {bog_path}")
    clean_replace_vlg(bog_tmp_path, bog_path)



def run_all_parallel(bench, design_name):
    design_js = "../../design_rtl_timer_pwr.json"
    with open(design_js, "r") as f:
        js = json.load(f)

    print(js[bench])

    param_lst = []
    if design_name == "":
        for design, design_vec in js[bench].items():
            top = design_vec[0]
            param_lst.append((bench, design, top))
        print(param_lst)
    else:
        for design, design_vec in js[bench].items():
            if design_name == design:
                top = design_vec[0]
                param_lst.append((bench, design, top))
    
    # run_one_design(param_lst[0])
    # exit()

    # design_lst = [d for d in js[bench]]

    # print(design_lst)
    # input()
    
    with Pool(20) as p:
        p.map(run_one_design, param_lst)
        p.close()
        p.join()


if __name__ == '__main__':
    global cmd
    cmd = "sog"
    cmd = "aig"
    cmd = "xag"
    cmd = "aimg"
    
    boom_lst = []

    
    bench_lst = [ "opencores", "itc", "VexRiscv", "riscvcores", "NVDLA", "chipyard"]
    # bench_lst = ["VexRiscv"]
    # bench_lst = ["itc"]
    # bench_lst = ["chipyard"]
    # design = "Rocket"
    design = "TinyRocket"

    for bench in bench_lst:
        run_all_parallel(bench, design)

