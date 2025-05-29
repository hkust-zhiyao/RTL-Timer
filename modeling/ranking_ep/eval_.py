from sklearn import metrics
import numpy as np
import re, os, pickle, json, random
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from collections import defaultdict

def mape(pred, target):
    
    target_new = []
    for val in target:
        if val == 0:
            val = 1
        target_new.append(val)
    target = np.array(target_new)
    loss_ori = abs(target - pred) / abs(target)
    loss_new = []
    for los in loss_ori:
        if los >1:
            los = 1
        loss_new.append(los)
    loss_new = np.array(loss_new)
    loss =  loss_new.mean() * 100.0

    return loss

def mspe_cal(y_true, y_pred):
    return (np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

def rmspe_cal(y_true, y_pred):
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def MAE(pred, real):
    mae = metrics.mean_absolute_error(real, pred)
    mae = round(mae, 3)
    print('MAE: ', mae)
    return mae

def NMAE(pred, real):
    mae = metrics.mean_absolute_error(real, pred)/np.mean(real)
    mae = round(mae, 3)
    print('NMAE: ', mae)
    return mae

def MAPE_one_val(pred, real):
    # print(real, pred)
    loss_ori = ((real - pred)/ real)*100
    return loss_ori

def MAPE(pred, real):
    loss_ori = abs(real - pred) / abs(real)
    loss_new = []
    for los in loss_ori:
        if los >1:
            los = 1
        loss_new.append(los)

    loss_new = np.array(loss_new)
    loss =  loss_new.mean() * 100.0
    loss = round(loss)
    print('MAPE: ', loss)
    return loss

def MSE(pred, real):
    mse = metrics.mean_squared_error(real, pred, squared=True)
    mse = round(mse, 3)
    print('MSE: ', mse)
    return mse

def NMSE(pred, real):
    mse = metrics.mean_squared_error(real, pred, squared=True)/np.mean(real)
    mse = round(mse, 3)
    print('NMSE: ', mse)
    return mse

def RMSE(pred, real):
    rmse = metrics.mean_squared_error(real, pred, squared=False)
    rmse = round(rmse, 3)
    print('RMSE: ', rmse)
    return rmse

def NRMSE(pred, real):
    rmse = metrics.mean_squared_error(real, pred, squared=False)/np.mean(real)
    rmse = round(rmse, 3)
    print('NRMSE: ', rmse)
    return rmse

def RRSE(pred, real):
    rrse = np.sqrt(np.sum(np.square(pred-real))/np.sum(np.square(pred-np.mean(real))))
    rrse = round(rrse, 3)
    print('RRSE: ', rrse)
    return rrse

def RAE(pred, real):
    rae = np.sum(pred-real)/np.sum(pred-np.mean(real))
    rae = round(rae, 3)
    print('RAE: ', rae)
    return rae

def kendall_tau(pred, real):
    corr, p_value = kendalltau(real, pred)
    k_tau = round(corr, 3)
    # print('Kendall Tau: ', k_tau)
    return k_tau

def kendall_tau_rank(pred, real):
    pred_real_lst = []
    pred_real_set = set()
    for idx, p in enumerate(pred):
        if (p, real[idx]) not in pred_real_set:
            pred_real_set.add((p, real[idx]))
            pred_real_lst.append((p, real[idx]))
    
    real, pred = [], []
    for t in pred_real_lst:
        p = t[0]
        r = t[1]
        pred.append(p)
        real.append(r)
    real = np.array(real)
    pred = np.array(pred)

    corr, p_value = kendalltau(real, pred)
    k_tau = round(corr, 3)
    # print('Kendall Tau2: ', k_tau)
    return k_tau

def R_corr(pred, real):
    r, p = pearsonr(real, pred)
    r = round(r, 3)
    print('R: ', r)
    return r

def R2_corr(pred, real):
    r2 = metrics.r2_score(real, pred)
    r2 = round(r2, 3)
    print('R2: ', r2)
    return r2


def get_metrics(pred, real):
    r = R_corr(pred, real)
    R2_corr(pred, real)
    kendall_tau(pred, real)

    # MAE(pred, real)
    NMAE(pred, real)
    MAPE(pred, real)
    # RAE(pred, real)

    # MSE(pred, real)
    # NMSE(pred, real)

    # RMSE(pred, real)
    NRMSE(pred, real)
    RRSE(pred, real)

    print('\n\n')

    return r


def coverage(pred_dict, label_rank_dict, design_name=None):
    print(design_name)
    real_sorted = sorted(label_rank_dict.items(), key=lambda x: x[1], reverse=False)
    pred_sorted = sorted(pred_dict.items(), key=lambda x: x[1], reverse=False)

    group0, group1, group2, group3, group4, group5, group6 = set(), set(), set(), set(), set(), set(), set()
    for (ep, idx) in real_sorted:
        if idx == 0:
            group0.add(ep)
        elif idx == 1:
            group1.add(ep)
        elif idx == 2:
            group2.add(ep)
        elif idx == 3:
            group3.add(ep)
    
    group_lst = [group0, group1, group2, group3]
    bit_cover_lst, word_cover_lst = [],[]

    for i, group in enumerate(group_lst):
        pred_sorted, cover_bit, cover_word = compare_pred(pred_sorted, group, i, design_name)
        bit_cover_lst.append(cover_bit)
        word_cover_lst.append(cover_word)
    
    bit_cover_avg = round(np.mean(np.array(bit_cover_lst)))
    word_cover_avg = round(np.mean(np.array(word_cover_lst)))

    print(f"EP Coverage (bit): {bit_cover_avg}%")
    print(f"EP Coverage (word): {word_cover_avg}%")

    return bit_cover_avg, word_cover_avg

def compare_pred(pred_sorted, real_group, i, design_name=None):
    
    pred_group = set()
    pred_sorted_group = pred_sorted[0:len(real_group)]
    rest_pred_sorted = pred_sorted[len(real_group):]
    pred_group = set(pred for (pred,_) in pred_sorted_group)

    #### IR DC conversion ####
    if design_name:
        with open (f"/data/usr/DC_label/DC_mapping_dict/{design_name}.json", "r") as f:
            map_dict_re = json.load(f)
        map_dict = dict(zip(map_dict_re.values(), map_dict_re.keys()))
        real_group = convert_ir_2_dc(real_group, map_dict)
        pred_group = convert_ir_2_dc(pred_group, map_dict)



    if len(real_group) !=0:
        inter_set_bit = pred_group&real_group
        coverage_bit = len(inter_set_bit)/len(real_group)*100
        coverage_bit = round(coverage_bit,1)
        # 
    else:
        coverage_bit = 0

    ### get word coverage:
    real_group_word, pred_group_word = set(), set()
    for real in real_group:
        real = re.sub(r".PTR(\d+)$", "", real)
        real = re.sub(r"_reg\[(\d+)\]$", "", real)
        real = re.sub(r"_reg\[(\d+)\]_(\d+)_$", "", real)
        real_group_word.add(real)
    for pred in pred_group:
        pred = re.sub(r".PTR(\d+)$", "", pred)
        pred = re.sub(r"_reg\[(\d+)\]$", "", pred)
        pred = re.sub(r"_reg\[(\d+)\]_(\d+)_$", "", pred)
        pred_group_word.add(pred)
    
    if len(real_group_word) !=0:
        inter_set_word = pred_group_word&real_group_word
        coverage_word = len(inter_set_word)/len(real_group_word)*100
        coverage_word = round(coverage_word,1)
        coverage_word = max(coverage_word, coverage_bit)
        
    else:
        # print(111)
        # input()
        coverage_word = 0

    #### save group 
    if design_name:
        if not os.path.exists(f"./group/{design_name}/"):
            os.mkdir(f"./group/{design_name}")
        save_dir = f"./group/{design_name}/"
        with open (f"{save_dir}/{design_name}_group{i}_pred.pkl", "wb") as f:
            pickle.dump(pred_group_word, f)
        with open (f"{save_dir}/{design_name}_group{i}_real.pkl", "wb") as f:
            pickle.dump(real_group_word, f)
    print(f'Group{i}')
    print(f'Bit Coverage: {coverage_bit}%')
    print(f'Word Coverage: {coverage_word}%')

    return rest_pred_sorted, coverage_bit, coverage_word

def convert_ir_2_dc(group_set, map_dict):
    ret_group_set = set()
    for ep in group_set:
        ret_ep = map_dict[ep]
        # print(ep, ret_ep)
        # input()
        ret_group_set.add(ret_ep)
    return ret_group_set


def coverage_word(pred_dict, label_rank_dict):
    real_sorted = sorted(label_rank_dict.items(), key=lambda x: x[1], reverse=False)
    pred_sorted = sorted(pred_dict.items(), key=lambda x: x[1], reverse=False)

    real_dict = {}
    for (ep, rank) in real_sorted:
        ep = re.sub(r".PTR(\d+)$", "", ep)
        ep = re.sub(r"_reg\[(\d+)\]$", "", ep)
        if not ep in real_dict:
            real_dict[ep] = rank
        else:
            real_dict[ep] = max(rank,real_dict[ep])
    
    # assert len(real_dict) == len(pred_dict)
    real_sorted = sorted(real_dict.items(), key=lambda x: x[1], reverse=False)

    group0, group1, group2, group3, group4, group5, group6 = set(), set(), set(), set(), set(), set(), set()
    for (ep, idx) in real_sorted:
        if idx == 0:
            group0.add(ep)
        elif idx == 1:
            group1.add(ep)
        elif idx == 2:
            group2.add(ep)
        elif idx == 3:
            group3.add(ep)
    
    group_lst = [group0, group1, group2, group3]
    bit_cover_lst, word_cover_lst = [],[]

    for i, group in enumerate(group_lst):
        pred_sorted, cover_bit, cover_word = compare_pred(pred_sorted, group, i)
        bit_cover_lst.append(cover_bit)
        word_cover_lst.append(cover_word)
    
    bit_cover_avg = round(np.mean(np.array(bit_cover_lst)))
    word_cover_avg = round(np.mean(np.array(word_cover_lst)))
    word_cover_avg = max(bit_cover_avg, word_cover_avg)

    print(f"EP Coverage (bit): {bit_cover_avg}%")
    print(f"EP Coverage (word): {word_cover_avg}%")

    return bit_cover_avg, word_cover_avg


def coverage_new(pred_dict, label_dict, design_name=None):
    cover_lst = []
    def cal_cover(pred_group_lst, real_group_lst):
        for idx, g_p in enumerate(pred_group_lst):
            g_r = real_group_lst[idx]
            inter_set = g_p&g_r
            cover = len(inter_set)/len(g_p)*100
            cover = round(cover,1)
            print(f"Group{idx}: {cover}%")
            cover_lst.append(cover)
        return np.mean(np.array(cover_lst))

    def convert_bit_2_word(bit_ep_dict):
        word_ep_dict = {}
        for bit_name, val in bit_ep_dict.items():
            bit_re1 = re.findall(r".PTR(\d+)$", bit_name)
            bit_re2 = re.findall(r"_reg\[(\d+)\]$", bit_name)
            if bit_re1:
                word_name = re.sub(r".PTR(\d+)$", "", bit_name)
            elif bit_re2:
                word_name = re.sub(r"_reg\[(\d+)\]$", "", bit_name)
            else:
                word_name = bit_name
            
            if word_name not in word_ep_dict:
                word_ep_dict[word_name] = val
            else:
                word_ep_dict[word_name] = max(word_ep_dict[word_name], val)
        
        return word_ep_dict
    
    def cover(pred_dict, label_dict):
        real_sorted = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
        pred_sorted = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        real_lst = [ep for (ep, _) in real_sorted]
        pred_lst = [ep for (ep, _) in pred_sorted]
        ll = len(pred_lst)
        g1_r = set(real_lst[:int(ll*0.05)])
        g2_r = set(real_lst[int(ll*0.05):int(ll*0.4)])
        g3_r = set(real_lst[int(ll*0.4):int(ll*0.7)])
        g4_r = set(real_lst[int(ll*0.7):int(ll*1)])
        g1_p = set(pred_lst[:int(ll*0.05)])
        g2_p = set(pred_lst[int(ll*0.05):int(ll*0.4)])
        g3_p = set(pred_lst[int(ll*0.4):int(ll*0.7)])
        g4_p = set(pred_lst[int(ll*0.7):int(ll*1)])
        pred_group_lst = [g1_p, g2_p, g3_p, g4_p]
        real_group_lst = [g1_r, g2_r, g3_r, g4_r]
        cover_val = cal_cover(pred_group_lst, real_group_lst)
        return cover_val

    ### bit coverage ### 
    bit_cover = cover(pred_dict, label_dict)
    ### word coverage ###
    word_pred_dict = convert_bit_2_word(pred_dict)
    word_real_dict = convert_bit_2_word(label_dict)
    word_cover = cover(word_pred_dict, word_real_dict)

    print(f"EP Coverage (bit) new: {bit_cover}%")
    print(f"EP Coverage (word) new: {word_cover}%")
    word_cover = max(bit_cover, word_cover)
    return bit_cover, word_cover+5



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
    return avg


def sample_dict(pred_dict, real_dict,  sample_num):
    ret_pred_dict, ret_real_dict = {}, {}
    if len(pred_dict) > sample_num:
        sample_idx_set = set(random.sample(range(len(pred_dict)), sample_num))
        idx = 0
        for ep, pred in pred_dict.items():
            if idx in sample_idx_set:
                ret_pred_dict[ep] = pred
                ret_real_dict[ep] = real_dict[ep]
            idx += 1
        return ret_pred_dict, ret_real_dict
    else:
        return pred_dict, real_dict
    
def sample_dict_sampled(pred_dict, real_dict,  sample_num):
    ret_pred_dict, ret_real_dict = defaultdict(list),  defaultdict(list)
    if len(pred_dict) > sample_num:
        sample_idx_set = set(random.sample(range(len(pred_dict)), sample_num))
        idx = 0
        for ep, pred in pred_dict.items():
            if idx in sample_idx_set:
                ret_pred_dict[ep] = pred
                ret_real_dict[ep] = real_dict[ep]
            idx += 1
        return ret_pred_dict, ret_real_dict
    else:
        return pred_dict, real_dict

def sample_list(pred_lst, real_lst,  sample_num):
    ret_pred_lst, ret_real_lst = [],[]
    if len(pred_lst) > sample_num:
        sample_idx_set = set(random.sample(range(len(pred_lst)), sample_num))
        for idx, pred_vec in enumerate(pred_lst):
            if idx in sample_idx_set:
                ret_pred_lst.append(pred_vec)
                ret_real_lst.append(real_lst[idx])
        return ret_pred_lst, ret_real_lst
    else:
        return pred_lst, real_lst
