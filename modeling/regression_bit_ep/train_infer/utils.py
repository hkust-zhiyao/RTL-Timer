import json
import torch
import numpy as np
from stat_ import MAPE_one_val, classify_metrics, regression_metrics

def get_all_label_cone(design_lst):
    ppa_dict = {}
    func_dict = {}
    for design in design_lst:

        # ============ PPA label ============
        ep_lst_path = f"/home/coguest5/rtl_repr/data_collect/label/ep_lst/{design}.json"
        with open (ep_lst_path, 'r') as f:
            ep_lst = json.load(f)
        
        ep_dict = {}

        for ep in ep_lst:
            js_path = f"/home/coguest5/rtl_repr/data_collect/label/ppa/cone_pwr_area/{design}/{ep}.json"
            with open (js_path, 'r') as f:
                ppa = json.load(f)
                ep_dict[ep] = ppa
        ppa_dict[design] = ep_dict

        # ============ Function label ============
        js_path = f"/home/coguest5/rtl_repr/data_collect/label/func/label_final/{design}.json"
        with open (js_path, 'r') as f:
            func = json.load(f)
        func_dict[design] = func
    # with open(f"./ppa_label.json", 'w') as f:
    #     json.dump(ppa_dict, f, indent=4)
    # with open(f"./func_label.json", 'w') as f:
    #     json.dump(func_dict, f, indent=4)

    return ppa_dict, func_dict



def lst_stat_ppa(lst):
    mean_r50 = np.mean(np.array(lst))
    mean_r40 = np.mean(np.array(lst[:40]))
    mean_r30 = np.mean(np.array(lst[:30]))
    mean_r20 = np.mean(np.array(lst[:20]))
    mean_r10 = np.mean(np.array(lst[:10]))
    mean_r5 = np.mean(np.array(lst[:5]))
    print(f"R@1: Avg: {lst[0]}, Std: {np.std(lst[0])}")
    print(f"R@5: Avg: {mean_r5}, Std: {np.std(lst[:5])}")
    print(f"R@10: Avg: {mean_r10}, Std: {np.std(lst[:10])}")
    print(f"R@20: Avg: {mean_r20}, Std: {np.std(lst)}")
    print(f"R@30: Avg: {mean_r30}, Std: {np.std(lst[:30])}")
    print(f"R@40: Avg: {mean_r40}, Std: {np.std(lst[:40])}")
    print(f"R@50: Avg: {mean_r50}, Std: {np.std(lst[:50])}")

    return lst[0], mean_r5, mean_r10, mean_r20, mean_r30, mean_r40, mean_r50

def lst_stat_func(lst):
    mean_r50 = np.mean(np.array(lst))
    mean_r40 = np.mean(np.array(lst[:40]))
    mean_r30 = np.mean(np.array(lst[:30]))
    mean_r20 = np.mean(np.array(lst[:20]))
    mean_r10 = np.mean(np.array(lst[:10]))
    mean_r5 = np.mean(np.array(lst[:5]))

    th = 0.3

    res_r50 = 1 if mean_r50 >= th else 0
    res_r40 = 1 if mean_r40 >= th else 0
    res_r30 = 1 if mean_r30 >= th else 0
    res_r20 = 1 if mean_r20 >= th else 0
    res_r10 = 1 if mean_r10 >= th else 0
    res_r5 = 1 if mean_r5 >= th else 0

    print(f"R@1: Res: {lst[0]}, mean: {lst[0]}")
    print(f"R@5: Res: {res_r5}, mean: {mean_r5}")
    print(f"R@10: Res: {res_r10}, mean: {mean_r10}")
    print(f"R@20: Res: {res_r20}, mean: {mean_r20}")
    print(f"R@30: Res: {res_r30}, mean: {mean_r30}")
    print(f"R@40: Res: {res_r40}, mean: {mean_r40}")
    print(f"R@50: Res: {res_r50}, mean: {mean_r50}")


    return lst[0], res_r5, res_r10, res_r20, res_r30, res_r40, res_r50





def func_stat(test_ep, pool_ep_lst, k):
    test_func = test_ep['func']
    pool_ep_lst_top_k = pool_ep_lst[:k]
    pool_func_lst = [ep['func'] for ep in pool_ep_lst_top_k]
    mean_pool = np.mean(np.array(pool_func_lst))

    th = 0.4
    res = 1 if mean_pool >= th else 0
    tf = True if res == test_func else False
    # print(f"R@{k} [Func] Real: {test_func} Pred: {res} TF: {tf}")

    return test_func, res

def ppa_stat(test_ep, pool_ep_lst, k):
    pool_ep_lst_top_k = pool_ep_lst[:k]

    ### slack ###
    test_slack = test_ep['ppa']['slack']
    pool_slack_lst = [ep['ppa']['slack'] for ep in pool_ep_lst_top_k]
    mean_pool_slack = round(np.mean(np.array(pool_slack_lst)), 2)
    std_pool_slack = round(np.std(pool_slack_lst), 2)
    mape_val = MAPE_one_val(test_slack, mean_pool_slack)
    # print(f"R@{k} [Slack] Real: {test_slack} Pred: {mean_pool_slack}, std: {std_pool_slack}, mape: {mape_val}")

    ### area ###
    test_area = test_ep['ppa']['area']
    pool_area_lst = [ep['ppa']['area'] for ep in pool_ep_lst_top_k]
    mean_pool_area = round(np.mean(np.array(pool_area_lst)),0)
    std_pool_area = round(np.std(pool_area_lst),0)
    mape_val = MAPE_one_val(test_area, mean_pool_area)
    # print(f"R@{k} [Area] Real: {test_area} Pred: {mean_pool_area}, std: {std_pool_area}, mape: {mape_val}")

    return test_slack, mean_pool_slack, test_area, mean_pool_area

    

def ppa_a_func_stat(test_ep, pool_ep_lst, k):
    ppa_stat(test_ep, pool_ep_lst, k)
    func_stat(test_ep, pool_ep_lst, k)


def stat_one_design_func(design_name, design_dct:dict):
    print(f"\n\n\nCurrent Design: {design_name}")
    real_func_lst = []
    pred_func_r1_lst, pred_func_r5_lst, pred_func_r10_lst, pred_func_r20_lst, pred_func_r30_lst, pred_func_r40_lst, pred_func_r50_lst = [], [], [], [], [], [], []
    min_dst_lst = []
    for ep, lst in design_dct.items():
        test_point = lst[0]
        pool_lst = lst[1:]

        if 'min_dst' in test_point:
            min_dst_lst.append(test_point['min_dst'])
        else:
            min_dst_lst.append(0)
        
        real, pred1 = func_stat(test_point, pool_lst, 1)
        real, pred5 = func_stat(test_point, pool_lst, 5)
        real, pred10 = func_stat(test_point, pool_lst, 10)
        real, pred20 = func_stat(test_point, pool_lst, 20)
        real, pred30 = func_stat(test_point, pool_lst, 30)
        real, pred40 = func_stat(test_point, pool_lst, 40)
        real, pred50 = func_stat(test_point, pool_lst, 50)

        real_func_lst.append(real)
        pred_func_r1_lst.append(pred1)
        pred_func_r5_lst.append(pred5)
        pred_func_r10_lst.append(pred10)
        pred_func_r20_lst.append(pred20)
        pred_func_r30_lst.append(pred30)
        pred_func_r40_lst.append(pred40)
        pred_func_r50_lst.append(pred50)

    ret_stat_dict = {}
    ret_stat_dict['confidence'] = np.mean(np.array(min_dst_lst))
    print('R@1')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r1_lst)
    ret_stat_dict['R@1'] = {"state":sen, "data":spe, "balance":balance}
    print('R@5')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r5_lst)
    ret_stat_dict['R@5'] = {"state":sen, "data":spe, "balance":balance}
    print('R@10')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r10_lst)
    ret_stat_dict['R@10'] = {"state":sen, "data":spe, "balance":balance}
    print('R@20')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r20_lst)
    ret_stat_dict['R@20'] = {"state":sen, "data":spe, "balance":balance}
    print('R@30')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r30_lst)
    ret_stat_dict['R@30'] = {"state":sen, "data":spe, "balance":balance}
    print('R@40')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r40_lst)
    ret_stat_dict['R@40'] = {"state":sen, "data":spe, "balance":balance}
    print('R@50')
    sen, spe, balance = classify_metrics(real_func_lst, pred_func_r50_lst)
    ret_stat_dict['R@50'] = {"state":sen, "data":spe, "balance":balance}

    return ret_stat_dict



def stat_one_design_ppa(design_name, design_dct:dict):
    print(f"\n\n\nCurrent Design: {design_name}")
    real_slack_lst, pred_slack_r1_lst, pred_slack_r5_lst, pred_slack_r10_lst, pred_slack_r20_lst, pred_slack_r30_lst, pred_slack_r40_lst, pred_slack_r50_lst = [], [], [], [], [], [], [], []
    real_area_lst, pred_area_r1_lst, pred_area_r5_lst, pred_area_r10_lst, pred_area_r20_lst, pred_area_r30_lst, pred_area_r40_lst, pred_area_r50_lst = [], [], [], [], [], [], [], []

    min_dst_lst = []
    for ep, lst in design_dct.items():
        test_point = lst[0]
        pool_lst = lst[1:]
        
        if 'min_dst' in test_point:
            min_dst_lst.append(test_point['min_dst'])

        real_slack, pred_slack1, real_area, pred_area1 = ppa_stat(test_point, pool_lst, 1)
        real_slack, pred_slack5, real_area, pred_area5 = ppa_stat(test_point, pool_lst, 5)
        real_slack, pred_slack10, real_area, pred_area10 = ppa_stat(test_point, pool_lst, 10)
        real_slack, pred_slack20, real_area, pred_area20 = ppa_stat(test_point, pool_lst, 20)
        real_slack, pred_slack30, real_area, pred_area30 = ppa_stat(test_point, pool_lst, 30)
        real_slack, pred_slack40, real_area, pred_area40 = ppa_stat(test_point, pool_lst, 40)
        real_slack, pred_slack50, real_area, pred_area50 = ppa_stat(test_point, pool_lst, 50)

        real_slack_lst.append(real_slack)
        pred_slack_r1_lst.append(pred_slack1)
        pred_slack_r5_lst.append(pred_slack5)
        pred_slack_r10_lst.append(pred_slack10)
        pred_slack_r20_lst.append(pred_slack20)
        pred_slack_r30_lst.append(pred_slack30)
        pred_slack_r40_lst.append(pred_slack40)
        pred_slack_r50_lst.append(pred_slack50)

        real_area_lst.append(real_area)
        pred_area_r1_lst.append(pred_area1)
        pred_area_r5_lst.append(pred_area5)
        pred_area_r10_lst.append(pred_area10)
        pred_area_r20_lst.append(pred_area20)
        pred_area_r30_lst.append(pred_area30)
        pred_area_r40_lst.append(pred_area40)
        pred_area_r50_lst.append(pred_area50)

    ret_stat_dict = {}
    ret_stat_dict['confidence'] = np.mean(np.array(min_dst_lst))
    print('R@1')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r1_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r1_lst, real_area_lst)
    ret_stat_dict['R@1'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@5')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r5_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r5_lst, real_area_lst)
    ret_stat_dict['R@5'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@10')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r10_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r10_lst, real_area_lst)
    ret_stat_dict['R@10'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@20')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r20_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r20_lst, real_area_lst)
    ret_stat_dict['R@20'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@30')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r30_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r30_lst, real_area_lst)
    ret_stat_dict['R@30'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@40')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r40_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r40_lst, real_area_lst)
    ret_stat_dict['R@40'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}
    print('R@50')
    r_s, mape_s, rrse_s = regression_metrics(pred_slack_r50_lst, real_slack_lst)
    r_a, mape_a, rrse_a = regression_metrics(pred_area_r50_lst, real_area_lst) 
    ret_stat_dict['R@50'] = {"slack": {"r":r_s, "mape":mape_s, "rrse":rrse_s}, "area": {"r":r_a, "mape":mape_a, "rrse":rrse_a}}

    return ret_stat_dict


# def eval_one_design_func(design, dct, model):
#     print(f"\n\n\nCurrent Design: {design}")
#     feat_lst_all, label_lst_all = [], []
#     for ep, lst in dct.items():
#         label_one_ep = lst[0]['func']
#         feat_one_ep = lst[-1]
#         feat_lst_all.append(feat_one_ep)
#         label_lst_all.append(label_one_ep)

#     feat_lst_all = np.array(feat_lst_all)
#     print(feat_lst_all.shape)
#     # feat_lst_all = feat_lst_all[:, :512]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     feat_lst_all = torch.from_numpy(feat_lst_all).to(torch.float32).to(device)
#     # pred_lst = model(feat_lst_all).argmax(dim=1).cpu().detach().numpy()
#     pred_lst_all = []
#     for feat_lst in feat_lst_all:
#         if feat_lst[0] > 5:
#             pred = int(feat_lst[257].view(1, -1).cpu().detach().numpy()[0][0])
#             pred_lst_all.append(pred)
#         else:
#             ### convert 1-d feat_lst to 2-d feat_lst ###
#             feat_lst = feat_lst.view(1, -1)
#             pred = model(feat_lst).argmax(dim=1).cpu().detach().numpy().tolist()[0]
#             pred_lst_all.append(pred)
    
#     classify_metrics(pred_lst_all, label_lst_all)
#     input()

def eval_one_design_func(design, dct, func_model):
    print(f"\n\n\nCurrent Design: {design}")
    feat_lst_all, label_lst_all = [], []
    for ep, lst in dct.items():
        label_one_ep = lst[0]['func']

        feat_one_ep = []
        feat_one_ep.extend(lst[0]['emb'])
        # feat_one_ep.extend(lst[1]['emb'])
        feat_one_ep.append(lst[1]['func'])
        feat_lst_all.append(feat_one_ep)
        label_lst_all.append(label_one_ep)

    feat_lst_all = np.array(feat_lst_all)
    print(feat_lst_all.shape)
    # feat_lst_all = feat_lst_all[:, :512]
    pred_lst = func_model.predict(feat_lst_all)
    classify_metrics(pred_lst, label_lst_all)
    input()


def eval_one_design_ppa(design, dct, area_model, slack_model):
    print(f"\n\n\nCurrent Design: {design}")
    feat_lst_all, label_lst_all_s, label_lst_all_a = [], [], []
    for ep, lst in dct.items():
        label_one_ep_s = lst[0]['ppa']['slack']
        label_one_ep_a = lst[0]['ppa']['area']
        feat_one_ep = []
        feat_one_ep.extend(lst[0]['emb'])
        # feat_one_ep.extend(lst[1]['emb'])
        feat_one_ep.append(lst[1]['ppa']['slack'])
        feat_lst_all.append(feat_one_ep)
        label_lst_all_s.append(label_one_ep_s)
        label_lst_all_a.append(label_one_ep_a)

    feat_lst_all = np.array(feat_lst_all)
    print(feat_lst_all.shape)
    # feat_lst_all = feat_lst_all[:, :512]
    # pred_lst_a = area_model.predict(feat_lst_all)
    pred_lst_s = slack_model.predict(feat_lst_all)
    regression_metrics(label_lst_all_s, pred_lst_s)
    # regression_metrics(label_lst_all_a, pred_lst_a)
    input()