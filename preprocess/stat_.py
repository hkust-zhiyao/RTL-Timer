from sklearn import metrics
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from collections import Counter

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
    # print('MAPE: ', round(loss_ori,1))
    return loss_ori

def MAPE(pred, real):
    loss_ori = abs(real - pred) / abs(real)
    ### if the real value is 0, set the loss to 0
    loss_ori = np.where(real == 0, 0, loss_ori)

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
    print('Kendall Tau: ', k_tau)
    return k_tau

def R_corr(pred, real):
    if len(pred) < 2:
        return 1
    r, p = pearsonr(real, pred)
    r = round(r, 3)
    print('R: ', r)
    return r

def R2_corr(pred, real):
    r2 = metrics.r2_score(real, pred)
    r2 = round(r2, 3)
    print('R2: ', r2)
    return r2


def regression_metrics(pred, real):
    

    pred = np.array(pred)
    real = np.array(real)

    r = R_corr(pred, real)
    mape_val = MAPE(pred, real)
    rrse_val = RRSE(pred, real)

    ### average prediction error
    mae = MAE(pred, real)

    # print('\n')

    return r, mape_val, rrse_val, mae


def classify_metrics(pred, real):
    pred = np.array(pred)
    real = np.array(real)



    recall = metrics.recall_score(real, pred)
    recall = round(recall, 3)
    print('sensitivity: ', recall)


    specificity = metrics.recall_score(real, pred, pos_label=0)
    specificity = round(specificity, 3)
    print('Specificity: ', specificity)

    balanced_accuracy = (recall + specificity)/2
    balanced_accuracy = round(balanced_accuracy, 3)
    print('Balanced Accuracy: ', balanced_accuracy)

    return recall, specificity, balanced_accuracy


def draw_scatter_plot(pred, real, save_dir, title=None):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(real, pred, alpha=0.5)
    max_val = max(max(pred), max(real))
    min_val = min(min(pred), min(real))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    # plt.plot([0, 1], [0, 1], 'r--')
    ## log scale
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Real')
    plt.ylabel('Pred')
    if title:
        plt.title(title)
    plt.show()
    plt.savefig(save_dir)


def draw_scatter_plot_color_bar(pred, real, node_attr, save_dir, title=None):
    ## draw scatter plot with node attribute value as color
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    import matplotlib.colorbar as colorbar

    fig, ax = plt.subplots()
    scatter = ax.scatter(pred, real, c=node_attr, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Node Attribute Value')
    ## log scale color bar
    norm = mcolors.LogNorm(vmin=min(node_attr), vmax=max(node_attr))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array(node_attr)
    plt.colorbar(sm, ax=ax, label='#. Gates')
    
    # Add reference line
    max_val = max(max(pred), max(real))
    min_val = min(min(pred), min(real))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    # plt.plot([0, 1], [0, 1], 'r--')
    ## log scale
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Real Power')
    plt.ylabel('Predicted Power')
    if title:
        plt.title(title)
    plt.show()
    plt.savefig(save_dir)
    plt.close()


def split_into_k_parts(lst, k, part_index):
    """Split lst into k approximately equal parts and return the part_index'th part"""
    n = len(lst)
    chunk_size = n // k
    remainder = n % k
    
    # Calculate start and end indices for the requested part
    start_idx = part_index * chunk_size + min(part_index, remainder)
    # Add one extra element to early chunks if there's remainder
    end_idx = start_idx + chunk_size + (1 if part_index < remainder else 0)
    
    return lst[start_idx:end_idx]


def coverage(pred_val_lst, real_val_lst, reg_lst, k=4):
    pred_dct, real_dct = {}, {}
    for idx, reg in enumerate(reg_lst):
        pred_dct[reg] = pred_val_lst[idx]
        real_dct[reg] = real_val_lst[idx]
    pred_dct_sorted = sorted(pred_dct.items(), key=lambda x: x[1], reverse=True)
    real_dct_sorted = sorted(real_dct.items(), key=lambda x: x[1], reverse=True)
    pred_lst, real_lst = [], []
    for i in range(len(pred_dct_sorted)):
        pred_lst.append(pred_dct_sorted[i][0])
        real_lst.append(real_dct_sorted[i][0])

    ## split the pred and real list into k parts
    pred_lst = np.array(pred_lst)
    real_lst = np.array(real_lst)
    cover_lst = []
    for i in range(k):
        pred = split_into_k_parts(pred_lst, k, i)
        real = split_into_k_parts(real_lst, k, i)
        pred_set = set(pred)
        real_set = set(real)
        inter_set = pred_set.intersection(real_set)
        if len(real_set) == 0:
            continue
        cover_i = len(inter_set) / len(real_set)
        cover_lst.append(cover_i)
    cover = round(np.mean(cover_lst), 2)*100
    print(f'Coverage: {cover}%')

    return cover


def coverage_rank_num(pred_scores, real_rank_lst):
    # Convert continuous scores to rank categories (0,1,2,3)
    # Sort scores and divide into quartiles
    scores_with_indices = [(i, score) for i, score in enumerate(pred_scores)]
    scores_with_indices.sort(key=lambda x: x[1])  # Sort by score (ascending)
    
    # Initialize ranks with -1
    pred_rank_lst = [-1] * len(pred_scores)
    
    # Calculate quartile sizes
    n = len(scores_with_indices)
    quartile_size = n // 4
    
    # Assign ranks (0,1,2,3)
    for rank in range(4):
        start_idx = rank * quartile_size
        end_idx = (rank + 1) * quartile_size if rank < 3 else n
        
        # Assign rank to each item in this quartile
        for i in range(start_idx, end_idx):
            idx, _ = scores_with_indices[i]
            pred_rank_lst[idx] = rank
    
    print(f'pred_rank_lst: {pred_rank_lst}')
    print(f'real_rank_lst: {real_rank_lst}')
    k = 4
    cover_lst = []
    for i in range(k):
        ## get the pred_rank and real_rank for the i-th part
        pred_rank = split_into_k_parts(pred_rank_lst, k, i)
        real_rank = split_into_k_parts(real_rank_lst, k, i)
        pred_counter = 0
        for pred in pred_rank:
            if pred == i:
                pred_counter += 1
        
        if len(real_rank) == 0:
            continue
        cover_i = pred_counter / len(real_rank)
        print(f'Coverage for rank {i}: {cover_i}')
        cover_lst.append(cover_i)
    cover = round(np.mean(cover_lst), 2)*100
    print(f'Coverage: {cover}%')
    return cover