import os, time, json
from multiprocessing import Pool
design_json = "/data/usr/LS-benchmark/design_timing_rgb.json"

def run_one_design(bench, design, cmd, clk, reset, top_name):
    bench_path = "/data/usr/LS-benchmark/{0}/{1}/".format(bench, cmd)
    design_dir = bench_path + design + '_' + cmd + '.v'
    print('Current Design: ', design)
    print('Current cmd: ', cmd)
    os.system(f'python3 analyze.py {design_dir} -N {design} -C {cmd} -K {clk} -R {reset} -T {top_name}')

def run_one_design_parrallel(design):
    bench = 'path'
    cmd = 'rtlil'
    bench_path = "/data/usr/LS-benchmark/{0}/{1}/".format(bench, cmd)
    design_dir = bench_path + design + '_' + cmd + '.v'
    print('Current Design: ', design)
    print('Current cmd: ', cmd)
    os.system(f'python3 analyze.py {design_dir} -N {design} -C {cmd}')

def run_all(bench, cmd, design_name=None):
    
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for k, v in bench_data.items():
        top_name = v[0]
        clk = v[1]
        reset = v[2]
        if design_name:
            if k == design_name:
                run_one_design(bench, k, cmd, clk, reset, top_name)
        else:
            run_one_design(bench, k, cmd, clk, reset, top_name)

def run_all_parallel(bench):
    
      with open(design_json, 'r') as f:
            design_data = json.load(f)
            bench_data = design_data[bench]
      with Pool(20) as p:
            p.map(run_one_design, list(bench_data.keys()))
            p.close()
            p.join()
    

if __name__ == '__main__':
    bench_list_all = ['itc','opencores','VexRiscv', 'chipyard', 'riscvcores','NVDLA']
    cmd = 'rtlil'
    design_name = ''

    
    for bench in bench_list_all:
        run_all(bench, cmd, design_name)

