import re, pickle
import sys, os
from pathlib import Path
folder = Path(__file__).parent.parent.parent
sys.path.append(str(folder))
from eval import *
from scipy.stats import stats
from draw_fig import draw_fig
from collections import defaultdict

class bit2word(object):
    def __init__(self, design_name, feat_dict):
        self.design_name = design_name
        self.feat_dict = feat_dict
        self.word_real_dict = {}
        self.word_pred_dict = {}
        self.bit_real_dict = {}
        self.bit_pred_dict = {}
        self.bit_feat_dict = {}
        self.b2w_map = defaultdict(set)
    
    def convert_bit_2_word(self, bit_name, pred, real):
        self.bit_pred_dict[bit_name] = pred
        self.bit_real_dict[bit_name] = real
        self.bit_feat_dict[bit_name] = self.feat_dict[bit_name]
        bit_re0 = re.findall(r"\[(\d+)\].PTR(\d+)$", bit_name)
        bit_re1 = re.findall(r".PTR(\d+)$", bit_name)
        bit_re2 = re.findall(r"_reg\[(\d+)\]$", bit_name)
        bit_re3 = re.findall(r"_reg\[(\d+)\]\[(\d+)\]$", bit_name)
        # if bit_re0:
        #     word_name = re.sub(r"\[(\d+)\].PTR(\d+)$", "", bit_name)
        if bit_re1:
            word_name = re.sub(r".PTR(\d+)$", "", bit_name)
        elif bit_re2:
            word_name = re.sub(r"_reg\[(\d+)\]$", "", bit_name)
        # elif bit_re3:
        #     word_name = re.sub(r"_reg\[(\d+)\]\[(\d+)\]$", "", bit_name)
        else:
            word_name = bit_name
        
        # if re.findall(r"\[(\d+)\]$", word_name):
        # if re.findall(r"PTR$", word_name):
        #     print(word_name)
        #     input()
        
        self.b2w_map[word_name].add(bit_name)
        

        if word_name not in self.word_pred_dict:
            self.word_pred_dict[word_name] = pred
            self.word_real_dict[word_name] = real
        else:
            self.word_pred_dict[word_name] = max(self.word_pred_dict[word_name], pred)
            self.word_real_dict[word_name] = max(self.word_real_dict[word_name], real)
    
    def get_word_stat(self):
        pred_lst = np.array(list(self.word_pred_dict.values()))
        real_lst = np.array(list(self.word_real_dict.values()))

        print(pred_lst.shape)
        print(real_lst.shape)

        r = draw_fig(self.design_name+".W", real_lst, pred_lst, 'sog')

        return r
    
    def get_bit_stat(self):
        pred_lst = np.array(list(self.bit_pred_dict.values()))
        real_lst = np.array(list(self.bit_real_dict.values()))
        r = draw_fig(self.design_name+".B", real_lst, pred_lst, 'sog')
        return r

    
    def save_dict(self, save_dir, cmd):
        if not os.path.exists(f"{save_dir}/{cmd}"):
            os.mkdir(f"{save_dir}/{cmd}")

        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_feat.pkl", "wb") as f:
            pickle.dump(self.bit_feat_dict, f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_pred.pkl", "wb") as f:
            pickle.dump(self.bit_pred_dict, f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_label.pkl", "wb") as f:
            pickle.dump(self.bit_real_dict, f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_word_pred.pkl", "wb") as f:
            pickle.dump(self.word_pred_dict, f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_word_label.pkl", "wb") as f:
            pickle.dump(self.word_real_dict, f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_b2w_map.pkl", "wb") as f:
            pickle.dump(self.b2w_map, f)
        
        print('Bit len: ', len(self.bit_real_dict))
        print('Word len: ', len(self.word_real_dict))
    
    def load_dict_for_training(self, save_dir, cmd):
        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_feat.pkl", "rb") as f:
            self.bit_feat_dict = pickle.load(f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_pred.pkl", "rb") as f:
            self.bit_pred_dict = pickle.load(f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_bit_label.pkl", "rb") as f:
            self.bit_real_dict = pickle.load(f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_word_pred.pkl", "rb") as f:
            self.word_pred_dict = pickle.load(f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_word_label.pkl", "rb") as f:
            self.word_real_dict = pickle.load(f)
        with open (f"{save_dir}/{cmd}/{self.design_name}_b2w_map.pkl", "rb") as f:
            self.b2w_map = pickle.load(f)

        
def avg_stat(lst, flag):
    avg = round(sum(lst)/len(lst),2)

    idx_1, idx_2, idx_3, idx_4=0,0,0,0 
    for r in lst:
        r = r/100
        if r >= 0.9:
            idx_1 += 1
        elif 0.9 > r >= 0.8:
            idx_2 += 1
        elif 0.8 > r >= 0.6:
            idx_3 += 1
        elif r < 0.6:
            idx_4 += 1
    
    print('\n')
    print(flag)
    print('Avg. R: ', avg)
    print('#. R>0.9: ', idx_1)
    print('#. 0.9>R>0.8: ', idx_2)
    print('#. 0.8>R>0.6: ', idx_3)
    print('#. R<0.6: ', idx_4)
    print('\n')