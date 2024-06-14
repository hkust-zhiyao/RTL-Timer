import os, re,json
design_json = "/home/coguest5/LS-benchmark/design_rtl_timer.json"

def clean_vlg(file):
    file_tmp = file + ".tmp"
    os.system(f"cp {file} {file_tmp}")
    os.remove(file)
    with open(file_tmp, "r") as f:
        lines = f.readlines()
        with open(file, "a+") as f_tmp:
            for line in lines:
                line = re.sub(r'\(\*(.*)\*\)', '', line)
                line = re.sub(r'/\*(.*)', '', line)
                if line.strip():
                    f_tmp.writelines(line)
    os.remove(file_tmp)

def clean_map_vlg(file, out_file=None):
    if not out_file:
        out_file = file
    file_tmp = file + ".tmp"
    os.system(f"cp {file} {file_tmp}")
    if out_file == file:
        os.remove(out_file)
    with open(file_tmp, "r") as f:
        lines = f.readlines()
        with open(out_file, "w+") as f_tmp:
            for line in lines:
                dff_re = re.findall(r'(DFF_X1|DFFRS_X1)(\s+)(\S+)(\s+)\(', line)

                line = re.sub(r'\(\*(.*)\*\)', '', line)
                line = re.sub(r'/\*(.*)', '', line)
                ### replace number ###
                line = re.sub(r'(\d+)\'h[0-9a-z]+', "1'b1", line)
                line = re.sub(r'(\d+)\'b[0-9a-z]+', "1'b1", line)
                ### replace DFF name ###
                if dff_re:
                    # print(line)
                    reg_name = dff_re[0][2]
                    reg_name = re.sub(r'\\', "", reg_name)
                    reg_name = re.sub(r'\$_(\w)*DFF(\w)*_(\S+)$', "", reg_name)
                    reg_name = re.sub(r'\.', "_", reg_name)
                    ps_re = re.findall(r'\[(\d+)\](\s*)\[(\d+)\]', reg_name)
                    if ps_re:
                        reg_name = re.sub(r'\[(\d+)\](\s*)\[(\d+)\]', r'_reg_{0}__{1}_'.format(ps_re[0][0], ps_re[0][2]), reg_name)
                    ptr_re = re.findall(r'\[(\d+)\]', reg_name)
                    if ptr_re:
                        reg_name = re.sub(r'\[(\d+)\]', r'_reg_{0}_'.format(ptr_re[0]), reg_name)
                    if not ps_re and not ptr_re:
                        reg_name = f'{reg_name}_reg'
                    
                    line = re.sub(r'(DFF_X1|DFFRS_X1)(\s+)(\S+)(\s+)\(', f'{dff_re[0][0]}{dff_re[0][1]}{reg_name}{dff_re[0][3]}(', line)

                if line.strip():
                    f_tmp.writelines(line)
    os.remove(file_tmp)

def run_all(bench, design_name=None):
    
    with open(design_json, 'r') as f:
        design_data = json.load(f)
        bench_data = design_data[bench]
    for name, v in bench_data.items():
        top_name = v[0]
        infile = f"/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/generated_netlist_file/{top_name}_{name}_TYP.syn.v"
        outfile = f"/home/coguest5/RTL-Timer/dataset/BOG/{cmd}/mapped_netlist/{top_name}_{name}_TYP.syn.v"
        if design_name:
            if name == design_name:
                print('Current Design: ', name)
                clean_map_vlg(infile, outfile)
        else:
            print('Current Design: ', name)
            clean_map_vlg(infile, outfile)


if __name__ == '__main__':
    global cmd
    cmd = 'SOG'

    bench_list_all = ['itc','opencores','VexRiscv', 'chipyard', 'riscvcores','NVDLA']
    design_name = ''

    for bench in bench_list_all:
        run_all(bench, design_name)
