
##### Group path option with endpoint register name ####
group_path -weight 5  -name user1  -to [find cell *P2/PhyAddrPointer*]
group_path -weight 5  -name user1  -to [find cell *P3/PhyAddrPointer*]
group_path -weight 5  -name user1  -to [find cell *P1/State2*]
group_path -weight 5  -name user1  -to [find cell *P3/State2*]
group_path -weight 5  -name user2  -to [find cell *P1/Datao*]
group_path -weight 5  -name user2  -to [find cell *P1/InstAddrPointer*]
group_path -weight 5  -name user2  -to [find cell *P2/State2*]
group_path -weight 5  -name user3  -to [find cell *P3/InstAddrPointer*]
group_path -weight 5  -name user3  -to [find cell *P2/InstAddrPointer*]
group_path -weight 5  -name user3  -to [find cell *P1/InstQueueRd_Addr*]
group_path -weight 5  -name user4  -to [find cell *P3/uWord*]
group_path -weight 5  -name user4  -to [find cell *P3/EAX*]
group_path -weight 5  -name user4  -to [find cell *P2/EAX*]


#### Retiming option with endpoint register name ####
set_dont_retime [get_cells *] true
set_dont_retime [get_cells {*P3/State2*}] false
set_dont_retime [get_cells {*P2/PhyAddrPointer*}] false
set_dont_retime [get_cells {*P2/State2*}] false
set_dont_retime [get_cells {*P1/InstAddrPointer*}] false
set_dont_retime [get_cells {*P1/Datao*}] false
set_dont_retime [get_cells {*P1/State2*}] false
set_dont_retime [get_cells {*P3/InstAddrPointer*}] false



