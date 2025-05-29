import os

def clean_folder(folder_path):
    ## Remove all json in the specified folder
    for filename in os.listdir(folder_path):
        design_name = filename.split('.')[0]
        if design_name == "TinyRocket":
            continue
        if "TinyRocket" in filename:
            continue
        os.remove(os.path.join(folder_path, filename))

def check_folder(folder_path):
    for filename in os.listdir(folder_path):
        design_name = filename.split('.')[0]
        print(f"Design: {design_name}")

if __name__ == "__main__":

    # for cmd in ['sog', 'aig', 'aimg', 'xag', 'net']:
    #     folder_path = f"/home/coguest5/RTL_PT_public/preprocess/rtl_timing_data/route/{cmd}"
    #     clean_folder(folder_path)

    folder_path = f"/home/coguest5/RTL_PT_public/RTL_timing_model/feat_label_en"
    clean_folder(folder_path)
