import re
import numpy as np

class CellNode:
    def __init__(self, name, tpe, driven_stren, trans, incr, arrival, polarity, fanout=None, cap=None):
        self.name = name
        self.tpe = tpe
        self.trans = trans
        self.incr = incr
        self.arrival = arrival
        self.driven_stren = driven_stren
        self.polarity = polarity
        self.fanout = fanout
        self.cap = cap

    def __repr__(self):
        return self.name

    def add_net_attribute(self, fanout, cap):
        self.fanout = fanout
        self.cap = cap

class NetNode:
    def __init__(self, name, fanout, cap):
        self.name = name
        self.tpe = 'net'
        self.fanout = fanout
        self.cap = cap

    def __repr__(self):
        return self.name

class timing_path(object):
    def __init__(self, start, end):
        self.start_end_pair = (start, end)
        # self.path = [start+'/CK', start+'/Q']
        self.path = []
        self.node_dict = {}
    
    def __repr__(self):
        return self.start_end_pair[0] + '__' + self.start_end_pair[1]

    def add_cell(self, line, cell_name=None):

        cell_line = re.findall(r"^  (\S+) \((\S+)\)(.*)", line)
        if re.findall(r"^  Point(\s+)Fanout(\s+)Cap(\s+)Trans(\s+)Incr(\s+)Path(.*)", line):
            return 
        elif re.findall(r"^  ----------", line):
            return
        elif re.findall(r"^  clock (\S+) \(rise edge\)", line):
            return
        elif re.findall(r"^  clock network delay \(propagated\)", line):
            return
        elif re.findall(r"^  clock network delay \(ideal\)", line):
            return
        elif cell_line:
            node_name = cell_line[0][0]
            node_cell = cell_line[0][1]
            node_num = cell_line[0][2]

            if node_cell != 'net':
                cell_re = re.findall(r"(\S+)\_X(\d+)", node_cell)
                node_tpe = cell_re[0][0]
                node_driven_stren = float(cell_re[0][1])
                self.path.append(node_name)
                num_re1 = re.findall(r"^(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)", node_num)
                trans = float(num_re1[0][1])
                incr = float(num_re1[0][3])
                arrival = float(num_re1[0][5])
                polarity = num_re1[0][7]
                cell_node = CellNode(node_name, node_tpe, node_driven_stren, trans, incr, arrival, polarity)
                self.node_dict[node_name] = cell_node
                return node_name
            else:
                num_re2 = re.findall(r"^(\s+)(\S+)(\s+)(\S+)", node_num)
                fanout = int(num_re2[0][1])
                cap = float(num_re2[0][3])
                if cell_name:
                    self.node_dict[cell_name].add_net_attribute(fanout, cap)

        elif re.findall(r"^  data arrival time", line):
            return    
        else:
            print(line)
            assert False
    
    def get_feat_from_lst(self, lst):
        lst = np.array(lst)
        ret1 = np.sum(lst)
        ret2 = np.mean(lst)
        ret3 = np.median(lst)
        ret4 = np.var(lst)
        ret5 = np.std(lst)
        

        return [ret1, ret2, ret3, ret4, ret5]
    
    def get_path_feat(self):
        delay = 0
        total_len = len(self.path)-3
        total_fanout = 0
        fanout_lst = []
        total_cap = 0
        cap_lst = []
        total_tran = 0
        tran_lst = []
        and_num, or_num, not_num, xor_num, mux_num, buf_num = 0,0,0,0,0,0
        
        for cell in self.path:
            delay = self.node_dict[cell].arrival
            if self.node_dict[cell].fanout:
                total_fanout += self.node_dict[cell].fanout
                fanout_lst.append(self.node_dict[cell].fanout)
            if self.node_dict[cell].cap:
                total_cap += self.node_dict[cell].cap
                cap_lst.append(self.node_dict[cell].cap)
            total_tran += self.node_dict[cell].trans
            tran_lst.append(self.node_dict[cell].trans)

            if self.node_dict[cell].tpe == 'AND2':
                and_num += 1
            elif self.node_dict[cell].tpe == 'OR2':
                or_num += 1
            elif self.node_dict[cell].tpe == 'INV':
                not_num += 1
            elif self.node_dict[cell].tpe == 'XOR2':
                xor_num += 1
            elif self.node_dict[cell].tpe == 'MUX2':
                mux_num += 1
            elif self.node_dict[cell].tpe == 'BUF':
                buf_num += 1
        # print(delay, total_len, total_fanout, total_cap, total_tran)
        # print(and_num, or_num, not_num, xor_num, mux_num, buf_num)

        

        ret_vec = [delay, total_len, \
                    and_num, or_num, not_num, xor_num, mux_num, buf_num]
        
        ret_lst = self.get_feat_from_lst(fanout_lst)
        ret_vec.extend(ret_lst)

        ret_lst = self.get_feat_from_lst(cap_lst)
        ret_vec.extend(ret_lst)

        ret_lst = self.get_feat_from_lst(tran_lst)
        ret_vec.extend(ret_lst)

        # print(ret_vec)

        return ret_vec
    

    #### used as embedding for EP-transformer
    def get_path_embedding(self):
        emb_lst = []
        for cell in self.path[1:-1]: ### without start-end registers
            tpe = self.node_dict[cell].tpe
            tpe = re.sub(r"(\d+)", "", tpe)

            fanout = self.node_dict[cell].fanout

            fanout = self.norm_fanout(fanout)

            cell_name = tpe + str(fanout)
            emb_lst.append(cell_name)
        
        return emb_lst

    

    def norm_fanout(self, fanout_num):
        ### 1-10, 15-95, 150-950, 1000 (29 numbers)
        if fanout_num <= 10:
            ret_num = fanout_num
        elif 10 < fanout_num <= 100:
            ret_num = int(fanout_num/10)*10+5
        elif 100 < fanout_num <= 1000:
            ret_num = int(fanout_num/100)*100+50
        else:
            ret_num = 1000

        return ret_num