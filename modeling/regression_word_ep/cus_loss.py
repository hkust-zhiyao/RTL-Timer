import numpy as np
def pseudo_huber_loss(y_pred, y_val):
    d = (y_val-y_pred)
    delta = 3
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

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