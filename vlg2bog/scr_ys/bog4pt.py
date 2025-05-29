import json, os




def run_all(bench, design_name):
    design_js = "../LS-benchmark/design_rtl_timer_pwr.json"
    with open(design_js, "r") as f:
        js = json.load(f)

    for design, design_vec in js[bench].items():
        top = design_vec[0]
        bog_path_old = f"../bog/{cmd}/{design}.{cmd}.v"
        bog_path_new = f"../bog_pt/{top}_{design}_TYP.{cmd}.v"
        os.system(f"cp {bog_path_old} {bog_path_new}")



if __name__ == '__main__':
    global cmd
    cmd = "sog"
    # cmd = "aig"
    # cmd = "xag"
    # cmd = "aimg"
    
    boom_lst = []

    
    bench_lst = [ "opencores", "itc", "VexRiscv", "riscvcores", "NVDLA", "chipyard"]
    # bench_lst = ["VexRiscv"]
    # bench_lst = ["itc"]
    # bench_lst = ["chipyard"]
    # design = "Rocket"
    design = ""

    for bench in bench_lst:
        run_all(bench, design)