# Annotating Slack Directly on Your Verilog: Fine-Grained RTL Timing Evaluation for Early Optimization

Wenji Fang, Shang Liu, Hongce Zhang, Zhiyao Xie. In Proceedings of ACM/IEEE Design Automation Conference (DAC), 2024.


## Abstract
In digital IC design, compared with post-synthesis netlists or layouts, the early register-transfer level (RTL) stage offers greater optimization flexibility for both designers and EDA tools. However, timing information is typically unavailable at this early stage. Some recent machine learning (ML) solutions propose to predict the total negative slack (TNS) and worst negative slack (WNS) of an entire design at the RTL stage, but the fine-grained timing information of individual registers remains unavailable. In this work, we address the unique challenges of RTL timing prediction and introduce our solution named RTL-Timer. To the best of our knowledge, this is the first fine-grained general timing estimator applicable to any given design. RTL-Timer explores multiple promising RTL representations and proposes customized loss functions to capture the maximum arrival time at register endpoints. RTL-Timer’s fine-grained predictions are further applied to guide optimization in a standard synthesis flow. The average results on unknown test designs demonstrate a correlation above 0.89, contributing around 3% WNS and 10% TNS improvement after optimization.


## Repo Structure

1. Register-oriented RTL processing 
    * Folder: register_oriented_processing
    * Preprocess the RTL data for timing modeling, including:
        * Convert RTL to pseudo netlist (folder: vlg2netlist), note that the original RTL is bit-blastted using Yosys.
        * Extract three levels of feature (folder: feature_extraction)
            * Design-level
            * Cone-level
            * Path-level

2. Timing Modeling
    * Folder: modeling
    * Customize loss function and explore different ML models to achieve both fine-grained register timing modeling and design WNS/TNS modeling
        * Register-bit regression (folder: regression_bit_ep) 
        * Register-word regression (folder: regression_word_ep)
        * Register ranking (folder: ranking)
        * Design regression (folder: regression_design_wns_tns)

3. Timing Optimization
    * Folder: optimization
    * We showcase an example of utilizing the predicted fine-grained arrival timing information on each register to enable 2 optimization options (i.e., **group_path** and **set_dont_retime**), in the TCL file "example_syn_option.tcl"


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