# Annotating Slack Directly on Your Verilog: Fine-Grained RTL Timing Evaluation for Early Optimization

Wenji Fang, Shang Liu, Hongce Zhang, Zhiyao Xie. In Proceedings of ACM/IEEE Design Automation Conference (DAC), 2024. [[paper]](https://zhiyaoxie.com/files/DAC24_RTLTimer.pdf)


## Abstract
In digital IC design, compared with post-synthesis netlists or layouts, the early register-transfer level (RTL) stage offers greater optimization flexibility for both designers and EDA tools. However, timing information is typically unavailable at this early stage. Some recent machine learning (ML) solutions propose to predict the total negative slack (TNS) and worst negative slack (WNS) of an entire design at the RTL stage, but the fine-grained timing information of individual registers remains unavailable. In this work, we address the unique challenges of RTL timing prediction and introduce our solution named RTL-Timer. To the best of our knowledge, this is the first fine-grained general timing estimator applicable to any given design. RTL-Timer explores multiple promising RTL representations and proposes customized loss functions to capture the maximum arrival time at register endpoints. RTL-Timer’s fine-grained predictions are further applied to guide optimization in a standard synthesis flow. The average results on unknown test designs demonstrate a correlation above 0.89, contributing around 3% WNS and 10% TNS improvement after optimization.


## Repo Structure
Note: Please change all the path to your current path first.

1. Register-oriented RTL processing 
    * Folder: register_oriented_processing
    * Preprocess the RTL data for timing modeling, including:
        * Convert RTL to pseudo netlist (folder: vlg2bog).
            - Convert RTL code into BOG netlist (e.g., SOG)
                ```
                $ cd ./register_oriented_processing/vlg2bog
                $ python3 auto_run.py
                ```
                - Input: Verilog
                - Output: /home/coguest5/RTL-Timer/dataset_example/BOG/SOG/ (mapped_netlist + SDC for STA)
            - Perform STA based on the generated SOG_netlist and SDC
                - Input: /home/coguest5/RTL-Timer/dataset_example/BOG/SOG/ (mapped_netlist + SDC for STA)
                - Output: BOG timing report (/home/coguest5/RTL-Timer/dataset_example/BOG/SOG/timing_rpt)
        
        * Extract path level of feature and label (folder: feature_extraction)
            - Get path feature and label pair
                ```
                $ cd ./register_oriented_processing/feature_extraction
                $ python3 get_ep_feat.py
                $ python3 get_ep_label.py
                ```
                - Input: BOG timing report (/home/coguest5/RTL-Timer/dataset_example/BOG/SOG/timing_rpt) + netlist timing report (/home/coguest5/RTL-Timer/dataset_example/netlist/netlist_rpt)
                - Output: bit-wise feature-label pair (/home/coguest5/RTL-Timer/modeling/feat_label/bit-wise/{design_name}.pkl)

2. Timing Modeling
    * Folder: modeling
    * Customize loss function and explore different ML models to achieve both fine-grained register timing modeling and design WNS/TNS modeling
        * Register-bit regression (folder: regression_bit_ep) 
            1. Most simple version: single representation (e.g., SOG) w/o ensemble learning + single critical path (slowest) w/o sampling 
                - training and inference
                ```
                $ cd ./modeling/regression_bit_ep/train_infer
                ## change training and testing design list in /home/coguest5/RTL-Timer/modeling/regression_bit_ep/train_infer/design_js
                $ python3 train.py
                $ python3 infer.py
                ```
                - Input: bit-wise feature-label pair
                - Output: saved model for testing designs /home/coguest5/RTL-Timer/modeling/regression_bit_ep/saved_model/bit_ep_model_{test_design}.pkl
            2. Enhanced version in paper: sample other paths
                - Other timing paths are obtained also in Prime Time: first specify the start and end registers, and then report the timing of the path
                - The customized loss function is implemented in folder "cus_loss", function "retain_max_every_n" in cus_loss.py

        * Register-word regression (folder: regression_word_ep) & ranking (folder: ranking_ep)
            - The bit-wise predictions from all four representations are concated as the feature for signal-wise modeling (ensemble learning)
        * Design regression (folder: regression_design_wns_tns)

3. Timing Optimization
    * Folder: optimization
    * We showcase an example of utilizing the predicted fine-grained arrival timing information on each register to enable 2 optimization options (i.e., **group_path** and **set_dont_retime**), in the TCL file "example_syn_option.tcl"

## Ealier version of our RTL-stage PPA modeling work: [MasterRTL (ICCAD'23)](https://github.com/hkust-zhiyao/MasterRTL)

## Citation
If RTL-Timer could help your project, please cite our work:

```
@inproceedings{fang2024annotating,
  title={Annotating Slack Directly on Your Verilog: Fine-Grained RTL Timing Evaluation for Early Optimization},
  author={Fang, Wenji and Liu, Shang and Zhang, Hongce and Xie, Zhiyao},
  booktitle={Proceedings of 2024 ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2024},
  organization={ACM}
}
```