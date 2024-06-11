from preprocess import *
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from lineartree import LinearBoostRegressor
from lineartree import LinearForestRegressor
from sklearn.ensemble import RandomForestRegressor

def run_xgb_label(x_train, y_train, x_test):

    xgbr = xgb.XGBRegressor(n_estimators=15, max_depth=4, nthread=25)

    xgbr.fit(x_train, y_train)


    importance = xgbr.feature_importances_
    print(importance)


    y_pred = xgbr.predict(x_test)
    y_pred2 = xgbr.predict(x_train)
    
    return y_pred, y_pred2

def kFold_train(x, y, title):
    kf = KFold(n_splits=9, shuffle=True)
    kf.split(x, y)
    test_lst = []
    pred_lst = []
    train_lst = []
    pred2_lst = []
    test_all_arr = np.array(test_lst)
    pred_all_arr = np.array(pred_lst)
    train_all_arr = np.array(train_lst)
    pred2_all_arr = np.array(pred2_lst)
    # r_lst, r2_lst, mse_lst, rmse_lst = 0,0,0,0
    for k, (train, test) in enumerate(kf.split(x,y)):
        print('Fold: ', k)
        x_train = x.iloc[train]
        x_test = x.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        y_pred, y_pred2 = run_xgb_label(x_train, y_train, x_test)
        test_all_arr = np.append(test_all_arr, np.array(y_test, dtype=np.float64))
        pred_all_arr = np.append(pred_all_arr, y_pred)
        train_all_arr = np.append(train_all_arr, np.array(y_train, dtype=np.float64))
        pred2_all_arr = np.append(pred2_all_arr, y_pred2)


    print(test_all_arr.shape)
    print(pred_all_arr.shape)

    for idx in range(len(pred_all_arr)):
        pred = pred_all_arr[idx]
        real = test_all_arr[idx]
        print(idx, (pred-real)/real, real)

    draw_fig_kf(title, test_all_arr, pred_all_arr, 'xgb_kf_test', 'Test')
    draw_fig_kf(title, train_all_arr, pred2_all_arr, 'xgb_kf_train', 'Train')





if __name__ == '__main__':
    flag = 'wns'
    label_data, feat_data = load_data(flag)
    x = feat_data
    y = label_data

    kFold_train(x, y, flag)

