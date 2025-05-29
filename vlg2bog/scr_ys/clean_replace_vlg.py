import os, re

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

def clean_replace_vlg(file_in, file_out):
    if os.path.exists(file_out):
        os.remove(file_out)
    with open(file_in, "r") as f:
        lines = f.readlines()
        with open(file_out, "a+") as f_tmp:
            for line in lines:
                line = re.sub(r'\(\*(.*)\*\)', '', line)
                line = re.sub(r'/\*(.*)', '', line)
                if line.strip():
                    print(line)
                    f_tmp.writelines(line)
    os.remove(file_in)

if __name__ == '__main__':
    file = "../bog/boom0_bog.v"
    clean_replace_vlg(file)

### replace gates ###
# line = re.sub(r"BUF_X1", "BUFFD0BWP", line)
# line = re.sub(r"AND2_X1", "AN2D0BWP", line)
# line = re.sub(r"OR2_X1", "OR2D0BWP", line)
# line = re.sub(r"XOR2_X1", "XOR2D0BWP", line)
# line = re.sub(r'MUX2_X1', 'MUX2D0BWP', line)
# line = re.sub(r'INV_X1', 'INVD0BWP', line)
# line = re.sub(r"DFFRS_X1", "DFCSND1BWP", line)
# line = re.sub(r"DFF_X1", "DFQD0BWP", line)