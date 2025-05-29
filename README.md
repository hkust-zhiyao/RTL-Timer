# RTL Dataset

- Example folder: vlg2bog/rtl_example

# Commercial EDA flow & label collection 

- Example folder: report_example/rpt_data/net
1. Conduct logic synthesis (& PR) on the RTL design and obtain PPA report as shown in the example folder
2. Parse PPA report 
    - Power report: report_example/parse_pwr_rpt_net.py
        ```
        // a. power label on netlist
        $ cd report_example/
        $ python3 parse_pwr_rpt_net.py
        // a. output: report_example/rtl_pwr_data/net_pwr_rpt
        ```
    - Timing report: report_example/parse_timing_rpt_net.py
        ```
        // b. timing label on netlist
        $ cd report_example/
        $ python3 parse_timing_rpt_net.py

        // b. output: report_example/rtl_timing_data
        ```
    These ground-truth reports are then aligned with BOG pseudo netlist reports


# RTL data preprocessing & ML model development

## Preprocessing
1. **Verilog2BOG** 
```
$ cd vlg2bog/scr_ys
// Input: Verilog file (LS-benchmark). Tool: Yosys+ABC
$ python3 auto_run_vlg2bog.py
$ python3 bog4pt.py
// Output: BOG pseudo netlist with NG45. File path: vlg2bog/bog_pt/{design_top}_{design_name}_TYP.{bog}.v
```
2. **Run STA on BOG pseudo netlist**
- Example folder: report_example/rpt_data/BOG/{bog}

```
// Input: BOG pseudo netlist. Tool: PrimeTime

// a. power feature on BOG
$ cd report_example/
$ python3 parse_pwr_rpt_bog.py
// a. output: report_example/rtl_pwr_data/bog_pwr_rpt

// b. timing feature on BOG
$ cd flow/bes_data/sta/scr/
$ python3 parse_timing_rpt_bog.py
$ python3 align_reg_name_BOG_to_net.py
// b. output: report_example/rtl_timing_data/bog_timing_rpt

// Output: aligned BOG & netlist statistics and PPA metrics. Soft link file path: preprocess/rtl_pwr_data & preprocess/rtl_timing_data
```

3. **Feature Engineering**
```
$ cd preprocess/
// Input: BOG-netlist aligned data
    // a. Power feature engineering
    $ python3 get_feat_label_pwr.py
    // a. Output: preprocess/feat_label_pwr/{design_name}_{bog}_{label_stage}.json
    // b. Timing feature engineering
    $ python3 get_feat_label_timing_slack.py // bit-level slack (preprocess/rtl_timing_data/{label_stage}) --> signal-level slack (preprocess/rtl_timing_data/{label_stage}_word)
    $ python3 get_feat_label_timing_slack.py
    // b. Output: preprocess/feat_label_timing/{design_name}_{bog}_{label_stage}_word.json
// Output: Feature & label for ML models
```
## ML model training & testing
1 Power model
```
$ cd RTL_pwr_model/
// Input: power feature & label 
$ python3 train_infer_k_fold_BOG.component.py // k-fold cross-validation, change the power and label type in the main function
// Output: Power prediction per module for combinational/register/clock network, aggregating for design-level total power
```
2 Timing model 
- Prediction model
```
$ cd RTL_timing_model/
// Input: timing feature & label 
$ python3 train_infer_k_fold_BOG.slack.py // k-fold cross-validation, change the slack label type in the main function
// Output: Timing slack prediction per register for both bit-level and signal-level, aggregating for design-level WNS/TNS
```
- Ranking model
```
$ cd RTL_timing_model/
// Input: timing feature & label (rank)
$ python3 train_infer_k_fold_BOG.rank.py // k-fold cross-validation
// Output: Timing slack rank prediction per register for both bit-level and signal-level, can be further used for timing optimization (folder: optimization)
```
