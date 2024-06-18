###################################################################

# Created by write_sdc on Sat Aug 12 02:24:27 2023

###################################################################
set sdc_version 2.1

set_units -time ns -resistance MOhm -capacitance fF -voltage V -current mA
set_operating_conditions -analysis_type on_chip_variation 
set_wire_load_mode top
set_wire_load_model -name 5K_hvratio_1_1 -library NangateOpenCellLibrary
set_max_transition 0.5 [current_design]
set_load -pin_load 0.2 [all_outputs]
set_load -min -pin_load 0.1 [all_outputs]
create_clock [get_ports clock]  -name CLK_clock  -period 0.5  -waveform {0 0.25}
set_clock_uncertainty -setup 0.15  [get_clocks CLK_clock]
set_clock_uncertainty -hold 0.1  [get_clocks CLK_clock]
group_path -weight 0.1  -name in2out  -from [list [get_ports clock] [get_ports reset] [all_inputs]] 
set_input_delay -clock CLK_clock  -max 0.05  [all_inputs]
set_output_delay -clock CLK_clock  -max 0.05  [get_ports all_outputs]
set_clock_groups  -asynchronous -name CLK_clock_others_1  -group [get_clocks   \
CLK_clock]
set_input_transition -max 0.2  [get_ports reset]
set_input_transition -min 0.2  [get_ports reset]
set_input_transition -max 0.2  [get_ports all_inputs]