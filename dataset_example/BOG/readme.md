Note: 
- For BOG netlist (get features)
    - upload netlist and sdc to vlsi_flow
    - use /home/coguest5/vlsi_flow/flow/bes_data/sta/scr/autoRun.py to perform STA
    - get timing report using /home/coguest5/vlsi_flow/flow/bes_data/sta/scr/get_timing_rpt.py
    - download timing report to folder /home/coguest5/RTL-Timer/dataset/BOG/SOG/timing_rpt
- For real synthesized netlist (get label)
    - use /home/coguest5/vlsi_flow/flow/bes_data/syn/scr/autoRun.py to perform synthesis
    - use /home/coguest5/vlsi_flow/flow/bes_data/sta/scr/autoRun.py to perform STA
    - get timing report using /home/coguest5/vlsi_flow/flow/bes_data/sta/scr/get_timing_rpt.py, and
    - - download timing report to folder /home/coguest5/RTL-Timer/dataset/netlist/netlist_rpt

## 1. BOG_netlist --> STA

cd /home/coguest5/RTL-Timer/dataset/BOG/SOG
scp -r ./mapped_netlist/* coguest5@acf3030.ece.ust.hk:/home/coguest5/vlsi_flow/flow/bes_data/syn/netlist/
scp -r ./generated_sdc_file/* coguest5@acf3030.ece.ust.hk:/home/coguest5/vlsi_flow/flow/bes_data/syn/sdc/



## 2. download STA rpt
cd /home/coguest5/RTL-Timer/dataset/BOG/SOG
scp -r coguest5@acf3030.ece.ust.hk:/home/coguest5/vlsi_flow/flow/bes_data/sta/scr/sog_rpt/* ./timing_rpt/

