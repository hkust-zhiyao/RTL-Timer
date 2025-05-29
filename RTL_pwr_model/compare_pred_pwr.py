import pickle
from stat_ import regression_metrics, draw_scatter_plot, draw_scatter_plot_color_bar
cmd = "sog"
cmd = "en"

label_cmd = "_init"
# label_cmd = "_route"

module_cmd = ""
# module_cmd = "_module"



with open (f"./saved_pwr/pwr_{cmd}_clk_dyn{label_cmd}_pred{module_cmd}.pkl", "rb") as f:
    clk_pwr_pred_lst = pickle.load(f)
with open (f"./saved_pwr/pwr_{cmd}_clk_dyn{label_cmd}_real{module_cmd}.pkl", "rb") as f:
    clk_pwr_real_lst = pickle.load(f)

with open (f"./saved_pwr/pwr_{cmd}_comb_dyn{label_cmd}_pred{module_cmd}.pkl", "rb") as f:
    comb_pwr_pred_lst = pickle.load(f)
with open (f"./saved_pwr/pwr_{cmd}_comb_dyn{label_cmd}_real{module_cmd}.pkl", "rb") as f:
    comb_pwr_real_lst = pickle.load(f)

with open (f"./saved_pwr/pwr_{cmd}_reg_dyn{label_cmd}_pred{module_cmd}.pkl", "rb") as f:
    reg_pwr_pred_lst = pickle.load(f)
with open (f"./saved_pwr/pwr_{cmd}_reg_dyn{label_cmd}_real{module_cmd}.pkl", "rb") as f:
    reg_pwr_real_lst = pickle.load(f)

total_pwr_pred_lst = []
total_pwr_real_lst = []
for i in range(len(clk_pwr_pred_lst)):
    total_pwr_pred_lst.append(clk_pwr_pred_lst[i] + comb_pwr_pred_lst[i] + reg_pwr_pred_lst[i])
    total_pwr_real_lst.append(clk_pwr_real_lst[i] + comb_pwr_real_lst[i] + reg_pwr_real_lst[i])

r_val, mape_val, rrse_val, mae_val = regression_metrics(total_pwr_pred_lst, total_pwr_real_lst)
print(f"Total Power Average")
draw_scatter_plot(total_pwr_pred_lst, total_pwr_real_lst, f"./fig/design_level/pwr_{cmd}_total{label_cmd}{module_cmd}.png", title=f"Total Power, R={round(r_val, 2)}, MAPE={round(mape_val, 1)}, RRSE={round(rrse_val, 3)}, MAE={round(mae_val, 3)}")
