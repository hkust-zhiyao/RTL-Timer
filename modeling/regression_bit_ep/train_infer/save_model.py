from preprocess import *
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from lineartree import LinearBoostRegressor
from lineartree import LinearForestRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def run_xgb_label(x_train, y_train):

    y_train = y_train.astype(float)
    xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=100, nthread=25)

    xgbr.fit(x_train, y_train)


    with open ("../saved_model/ep_model_sog.pkl", "wb") as f:
        pickle.dump(xgbr, f)




if __name__ == '__main__':
    label_data, feat_data = load_data()
    x = feat_data
    y = label_data ## seq label

    # kFold_train(x, y, 'EP Model')
    run_xgb_label(x, y)
