rm -rf rtl_pwr_data*
scp -r coguest5@acf3030.ece.ust.hk:/home/coguest5/vlsi_flow_bog/rtl_pwr_data.tar.gz ./
tar -zxvf rtl_pwr_data.tar.gz 



scp -r coguest5@acf3030.ece.ust.hk:/home/coguest5/vlsi_flow_bog/flow/bes_data/sta/scr/bog_timing_aligned_rpt.tar.gz ./
tar -zxvf bog_timing_aligned_rpt.tar.gz
mv bog_timing_aligned_rpt rtl_timing_data