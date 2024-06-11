import numpy as np
import copy
def pseudo_huber_loss(y_pred, y_val):
    d = (y_val-y_pred)
    delta = 5  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def retain_max_every_n(src_arr, dst_arr, n):
    # Ensure the array can be reshaped into (-1, n)
    if src_arr.size % n != 0:
        raise ValueError(f"Array size ({src_arr.size}) should be divisible by n ({n})")
    
    # Reshape the array
    reshaped_src = src_arr.reshape(-1, n)
    reshaped_dst = dst_arr.reshape(-1, n)
    
    # Get indices of max values along axis 1 (columns)
    reshaped_abs = np.abs(reshaped_src)
    max_indices = np.argmax(reshaped_abs, axis=1)
    
    # Create a mask of zeros and set ones at the max_indices positions
    mask = np.zeros_like(reshaped_src, dtype=bool)
    mask[np.arange(reshaped_src.shape[0]), max_indices] = True
    
    # Apply the mask to the reshaped array
    reshaped_dst[~mask] = 0
    
    # Reshape back to 1D
    result = reshaped_dst.reshape(-1)
    
    return result


def retain_max_every_n_smooth(src_arr, dst_arr, n):
    # Ensure the array can be reshaped into (-1, n)
    if src_arr.size % n != 0:
        raise ValueError(f"Array size ({src_arr.size}) should be divisible by n ({n})")
    
    # Reshape the array
    reshaped_src = src_arr.reshape(-1, n)
    reshaped_dst = dst_arr.reshape(-1, n)
    
    # Get indices of max values along axis 1 (columns)
    reshaped_abs = np.abs(reshaped_src)
    max_val = np.max(reshaped_abs, axis=1)
    
    # Create a mask of zeros and set ones at the max_indices positions
    mask = copy.copy(reshaped_src)
    mask = mask/mask.max(axis=1).reshape(1,-1).T
    weight = np.array([1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1])
    weight = np.tile(weight, (mask.shape[0],1))
    # mask[np.arange(reshaped_src.shape[0]), max_indices] = True
    
    # Apply the mask to the reshaped array
    reshaped_dst = reshaped_dst*mask*weight
    
    # Reshape back to 1D
    result = reshaped_dst.reshape(-1)
    
    return result


def pseudo_huber_loss_max(y_real, y_pred):
    d = (y_pred-y_real)
    d = retain_max_every_n(y_pred, d, 16)
    delta = 5  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def retain_max(arr, repeat_num):
    # Reshape the array into (-1, 3)
    reshaped = arr.reshape(-1, repeat_num)
    
    # For each row, set all values to zero except the max
    for row in reshaped:
        max_index = np.argmax(row)
        for i in range(repeat_num):
            if i != max_index:
                row[i] = 0
                
    # Reshape back to 1-D array
    return reshaped.ravel()

def pseudo_huber_repeat(y_pred, y_val):
    d = (y_val-y_pred)
    d = retain_max(d, 3)
    delta = 5  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def pseudo_huber_loss_dm(y_pred, y_val):
    y_val = y_val.get_label()
    d = (y_val-y_pred)
    delta = 5  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def group_loss(y_pred, dm):
    group_lst = dm.get_group()
    y_val = dm.get_label()
    grad_lst, hess_lst = [], []

    for grp in group_lst:
        y_pred_grp = y_pred[0:grp]
        y_pred = y_pred[grp:]
        y_val_grp = y_val[0:grp]
        y_val = y_val[grp:]
        grad, hess = pseudo_huber_loss(y_pred_grp, y_val_grp)
        grad_lst.append(grad.tolist())
        hess_lst.append(hess.tolist())

    grad = np.average(np.array(grad_lst))
    hess = np.mean(np.array(hess_lst))

    return grad, hess




def range_loss(y_pred, y_true):
    max_val = 0.7
    min_val = 0.3
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def inverse_sigmoid(y, min_value, max_value):
        return np.log(y - min_value) - np.log(max_value - y)

    # def constrained_objective(y_true, y_pred, min_value, max_value):
    #     # Transform y_true to match the scale of the transformed predictions
    #     transformed_y_true = inverse_sigmoid(y_true, min_value, max_value)
        
    #     # Compute the residuals
    #     residuals = y_pred - transformed_y_true
        
    #     # Gradient and hessian for MSE
    #     gradient = 2 * residuals
    #     hessian = 2 * np.ones_like(residuals)
        
    #     return gradient, hessian

    def constrained_objective(y_true, y_pred, min_value, max_value):
        # Transform y_true to match the scale of the transformed predictions
        # transformed_y_true = inverse_sigmoid(y_true, min_value, max_value)
        residuals = y_true - y_pred
    
        # Compute the penalty for predictions outside the range
        lower_penalty = min_value - y_pred
        upper_penalty = y_pred - max_value
        
        # Apply penalties

        print(residuals)
        residuals += np.where(y_pred < min_value, lower_penalty, 0)
        residuals += np.where(y_pred > max_value, upper_penalty, 0)
        print(residuals)
        # input()
        d = residuals
        delta = 5  
        scale = 1 + (d / delta) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt 
        hess = (1 / scale) / scale_sqrt
        
        return grad, hess

    return constrained_objective(y_true, y_pred, min_val, max_val)

def weighted_objective(y_pred, y_true):
    # Compute the residuals
    residuals = y_pred - y_true
    
    # Weight the residuals by the labels
    weighted_residuals = residuals * y_true
    
    # Gradient and hessian for weighted squared error loss
    gradient = 2 * weighted_residuals
    hessian = 2 * np.abs(y_true)  # The hessian is weighted by the absolute value of the label
    
    return gradient, hessian

def assym_loss(y_val, y_pred):
    grad = np.where((y_val - y_pred)<0, -2*500.0*(y_val - y_pred), -2*(y_val - y_pred))
    hess = np.where((y_val - y_pred)<0, 2*500.0, 2.0)
    return grad, hess