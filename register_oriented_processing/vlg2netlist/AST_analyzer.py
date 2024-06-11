import copy, sys, time, json
from DG import *


class AST_analyzer(object):
    def __init__(self, ast, clk, reset):
        self.__ast = ast
        self.graph = Graph()
        self.oper_label = 0
        self.const_label = 0
        self.wire_set = set()

        self.wire_dict = {}
        self.temp_dict = {}
        self.func_dict = {}
        
        self.clk = clk
        self.reset = reset
        self.decl_node_dict = {}
        self.reg_assign_dict = {}
        self.netlist_cell_dict = {}
        self.netlist_cell_label = 0
        self.line_lst = []
        self.decl_set = set()
        self.name_map = {}
        self.name_map2 = {}

    def AST2Graph(self, ast, file_path):
        self.get_module_line(file_path)

        self.traverse_AST(ast)
        
        if self.reset:
            line = f"  INV_X1 U{self.netlist_cell_label} ( .A({self.reset}), .ZN(reset_n_DEFINE));\n"
            self.line_lst.append(line)
        self.line_lst.append("endmodule")
        print(len(self.line_lst))

    def get_module_line(self, file_path):
        with open (file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.findall(r'^module (.*)', line):
                self.line_lst.append(line)
                if self.reset:
                    self.line_lst.append("  wire reset_n_DEFINE;\n")
                return 


       
    def traverse_AST(self, ast):
        node_type = ast.get_type()
        if node_type == 'Decl':
            self.add_decl_node(ast, node_type)
        elif node_type == 'Always':
            self.add_reg(ast)
            self.write_reg_assign()
        elif node_type == 'Assign':
            self.add_wire_assign(ast)
            
        
        for c in ast.children():
            self.traverse_AST(c)

    
    def add_reg(self, ast):
        if ast.get_type() == 'NonblockingSubstitution':
            Lval = ast.left
            Rval = ast.right
            assert (Lval.get_type() == 'Lvalue') and (Rval.get_type() == 'Rvalue')
            LHS = self.add_new_node(Lval.var)
            RHS = self.add_new_node(Rval.var)
            D = RHS
            Q = LHS
            self.reg_assign_dict['D'] = D
            self.reg_assign_dict['Q'] = Q
            
        for c in ast.children():
            self.add_reg(c)
    
    def write_reg_assign(self):
        D = self.reg_assign_dict['D']
        Q = self.reg_assign_dict['Q']
        D = self.legalize_name(D)
        Q = self.legalize_name(Q)
        if self.reset:
            line = f"  DFFR_X1 {Q} ( .D({D}), .CK({self.clk}), .RN(reset_n_DEFINE), .Q({Q}), .QN());\n"
        else:
            line = f"  DFF_X1 {Q} ( .D({D}), .CK({self.clk}), .Q({Q}), .QN());\n"
        self.line_lst.append(line)
        self.reg_assign_dict.clear()
        

    def add_wire_assign(self, ast):
        assert len(ast.children()) == 2 ## only lest_var and right_var
        Lval = ast.left
        Rval = ast.right
        assert (Lval.get_type() == 'Lvalue') and (Rval.get_type() == 'Rvalue')
        LHS = self.add_new_node(Lval.var)
        RHS = self.add_new_node(Rval.var)

        

        if LHS == 'Concat':
            return
        if self.check_PS([LHS, RHS]):
            return
        if self.graph.node_dict[LHS].width != 1:
            return
        Z = self.legalize_name(LHS)
        
        if RHS in self.graph.node_dict:
            A = self.legalize_name(RHS)
            tpe = 'BUF_X1'
            name = f'U{self.netlist_cell_label}'
            self.netlist_cell_label += 1
            line = f"  {tpe} {name} ( .A({A}), .Z({Z}) );\n"
            self.line_lst.append(line)
            self.netlist_cell_dict.clear()
        elif RHS in ['INV_X1']:
            A = self.add_new_node(Rval.var.right)
            A = self.legalize_name(A)
            tpe = RHS
            name = f'U{self.netlist_cell_label}'
            self.netlist_cell_label += 1
            line = f"  {tpe} {name} ( .A({A}), .ZN({Z}) );\n"
            self.line_lst.append(line)
            self.netlist_cell_dict.clear()
        elif RHS in ['AND2_X1', 'OR2_X1', 'XOR2_X1']:
            A1 = self.add_new_node(Rval.var.left)
            A2 = self.add_new_node(Rval.var.right)
            A1 = self.legalize_name(A1)
            A2 = self.legalize_name(A2)
            tpe = RHS
            name = f'U{self.netlist_cell_label}'
            self.netlist_cell_label += 1
            if RHS in ['XOR2_X1']:
                line = f"  {tpe} {name} ( .A({A1}), .B({A2}), .Z({Z}) );\n"
            else:
                line = f"  {tpe} {name} ( .A1({A1}), .A2({A2}), .ZN({Z}) );\n"
            self.line_lst.append(line)
            self.netlist_cell_dict.clear()
        elif RHS in ['MUX2_X1']:
            S = self.add_new_node(Rval.var.cond)
            A = self.add_new_node(Rval.var.false_value)
            B = self.add_new_node(Rval.var.true_value)
            S = self.legalize_name(S)
            A = self.legalize_name(A)
            B = self.legalize_name(B)
            tpe = RHS
            name = f'U{self.netlist_cell_label}'
            self.netlist_cell_label += 1
            line = f"  {tpe} {name} ( .A({A}), .B({B}), .S({S}), .Z({Z}) );\n"
            self.line_lst.append(line)
            self.netlist_cell_dict.clear()
        elif RHS in ['Concat', "1'b0", "1'b1"]:
            return
        else:
            print(LHS)
            print(RHS)
            assert False


    def check_PS(self, name_lst):
        for name in name_lst:
            if re.findall('.PS(\d+)_(\d+)$', name):
                return True
        return False
    
    def legalize_name(self, name):
        name_ori = copy.copy(name)
        name = re.sub(r'\\', '', name)
        name_re = re.findall('(.*)\[(\d+)\](.*)', name)
        while name_re:
            name = name_re[0][0] + '_PTR' + str(name_re[0][1]) + name_re[0][2]
            name_re = re.findall('(.*)\[(\d+)\](.*)', name)

        # if name_re:
        #     name = name_re[0][0] + '_PTR' + str(name_re[0][1]) + name_re[0][2]
        name = re.sub(r'\.', '_', name)
        self.name_map[name_ori] = name
        return name




    def add_parent_edge(self):
        for name, node in self.graph.node_dict.items():
            if node.father:
                if self.graph.node_dict[node.father].type == 'Reg':
                    self.graph.add_edge(node.father, name)

    def func2graph(self, ast, input_list):
        nodetype = ast.get_type()
        if nodetype in ['Input']:
            input_list.append(str(ast.name))
        for c in ast.children():
            self.func2graph(c, input_list)

    def analyze_function(self):
        for f in self.func_set.copy():
            input_list = []
            assign_list = []
            self.func2graph(f, input_list)
            func_list = [input_list, f]
            self.func_dict[str(f.name)] = func_list
            self.func_set.remove(f)
  
    def add_decl_node(self, ast, node_type):
        if node_type == 'Decl':
            ll = len(ast.children())
            child = ast.children()[0]
            child_type = child.get_type()
            name = child.name
            width = self.get_width(child)
            self.graph.add_decl_node(name, child_type, width, None)
            
            name = self.legalize_name(name)

            if name in self.decl_set:
                return
            else:
                self.decl_set.add(name)

            tpe = child_type.lower()
            if width == 1:
                line = f"  {tpe} {name};\n"
            else:
                width -= 1
                line = f"  {tpe} [{width}:0] {name};\n"
            
            if tpe not in ['reg']:
                self.line_lst.append(line)
            self.decl_node_dict.clear()

            
            # if ll == 2:
            #     child2 = ast.children()[1]
            #     self.add_assign_edge(child2)
            #     assert False

            

    def cal_width(self, ast):
        msb = int(ast.msb.value)
        lsb = int(ast.lsb.value)
        LHS = max(msb, lsb)
        RHS = min(msb, lsb)
        width = LHS - RHS + 1
        return width

    def get_width(self, ast): # -> int
        width = ast.width
        dimens = ast.dimensions
        if width:
            width = self.cal_width(width)
        else:
            width = 1
        if dimens:
            length = dimens.lengths[0]
            length = self.cal_width(length)
        else:
            length = 1
        return width*length

    def add_assign_edge(self, ast, node_type=None, sub_dict=None):
        node_type = ast.get_type()
        
        ### directly assign
        if node_type in ['Assign',  'NonblockingSubstitution', 'BlockingSubstitution']:
            self.add_assign(ast)
        elif node_type == 'Block':
            ast_tuple = ast.statements
            for ast in ast_tuple:
                self.add_assign_edge(ast)
        ### nested if statement
        elif node_type == 'IfStatement':
            cond = ast.cond
            ts = ast.true_statement
            fs = ast.false_statement
            mux_name = 'Mux' + str(self.oper_label)
            self.oper_label += 1
            cond_width = self.get_node_width(cond)
            if not cond_width:
                if cond.get_type() == 'Concat':
                    cond_width = 2
            self.graph.add_decl_node(mux_name, 'Operator', cond_width, None)
            # add mux
            # 1. if without else -> no mux
            if not fs:
                self.add_assign(ts)
            elif fs.get_type() != 'NonblockingSubstitution':
                self.assign(cond, mux_name)
                LHS1 = self.add_assign(ts, mux_name)
                self.graph.add_edge(LHS1, mux_name)

            # 2. if with else -> one mux
            elif (ts != fs) and (ts.get_type() == fs.get_type()):
                self.assign(cond, mux_name)
                LHS0 = self.add_assign(ts, mux_name)
                LHS1 = self.add_assign(fs, mux_name)
                assert LHS0 == LHS1
                self.graph.add_edge(LHS0, mux_name)


            # 3. if else if -> multiple mux
            else:
                self.add_assign_edge(fs)
        ### nested case statement
        elif node_type in ['CaseStatement', 'CasezStatement', 'CasexStatement', 'UniqueCaseStatement']:
            cond = ast.comp
            mux_name = 'Mux' + str(self.oper_label)
            self.oper_label += 1
            self.assign(cond, mux_name)
            cond_width = self.get_node_width(cond)
            if not cond_width:
                if cond.get_type() == 'Concat':
                    cond_width = 2
            self.graph.add_decl_node(mux_name, 'Operator', cond_width, None)
            caselist = ast.caselist
            for case_assign in caselist:
                self.add_case_assign(case_assign, cond_width)

    
    def add_case_assign(self, ast, width):
        child = ast.children()
        ll = len(child)
        if ll >= 2:
            cond = child[0]
            sta = ast.statement
            mux_name = 'Mux' + str(self.oper_label)
            self.oper_label += 1
            self.assign(cond, mux_name)
    
            self.graph.add_decl_node(mux_name, 'Operator', width, None)
            LHS1 = self.add_assign(sta, mux_name)
            self.graph.add_edge(LHS1, mux_name)
        elif ll == 1:
            pass
            ### TODO: event statement
        else:
            print(ll)
            print(child)
            assert False



    def get_node_width(self, ast):
        node_type = ast.get_type()
        parent_type = ast.get_parent_type()
        
        if node_type == 'Identifier':
            width = self.graph.node_dict[ast.name].width
        elif node_type == 'Pointer':
            width = 1
        elif node_type == 'Partselect':
            self.add_new_node(ast)
            width = self.graph.node_dict[ast.var.name].width
        elif node_type == 'IntConst':
            width = self.get_width_num(ast.value)
        elif node_type in ['Concat']:
            width = None
        elif parent_type == 'UnaryOperator':
            width = self.get_node_width(ast.right)
        else:
            print(node_type)
            assert False

        return width
    def add_assign(self, ast, L=None):
        node_type = ast.get_type()
        if node_type == 'IfStatement':
            self.add_assign_edge(ast)
            return
        elif node_type == 'Block':
            ast_tuple = ast.statements
            for ast in ast_tuple:
                self.add_assign_edge(ast)
            return
        elif node_type in ['CaseStatement', 'CasezStatement', 'CasexStatement', 'UniqueCaseStatement']:
            self.add_assign_edge(ast)
            return
        elif node_type == 'EventStatement':
            return
        assert node_type in ['Assign', 'NonblockingSubstitution', 'BlockingSubstitution']
        Lval = ast.left
        Rval = ast.right
        assert (Lval.get_type() == 'Lvalue') and (Rval.get_type() == 'Rvalue')
        
        if Lval.var.get_type() == 'LConcat':
            for LH in self.add_new_node(Lval.var):
                self.assign(Rval.var, LH)
        
        else:
        
            LHS = self.add_new_node(Lval.var)
            if not L:
                self.assign(Rval.var, LHS)
            else:
                self.assign(Rval.var, L)
            return LHS

            
    def add_new_node(self, ast):
        node_type = ast.get_type()
        parent_type = ast.get_parent_type()
        if node_type == 'Identifier':
            node_name = ast.name
            assert node_name in self.graph.node_dict.keys()
        elif node_type == 'Pointer':
            name = ast.var.name
            ptr = ast.ptr.value
            node_name = name + '_PTR' + ptr
            self.name_map2[node_name] = name + '.PTR' + ptr
            if node_name not in self.graph.node_dict.keys():
                self.graph.add_decl_node(node_name, 'Pointer', 1, name)
        elif node_type == 'Partselect':
            name = ast.var.name
            if (ast.msb.get_type() != 'IntConst' or ast.msb.get_type() != 'IntConst'):
                node_name = name
            else:
                msb = ast.msb.value
                lsb = ast.lsb.value
                width = self.cal_width(ast)
                node_name = name + '.PS' + msb + '_' + lsb
                if node_name not in self.graph.node_dict.keys():
                    self.graph.add_decl_node(node_name, 'Partselect', width, name)
        elif node_type == 'LConcat':
            node_name = 'Concat'
        elif node_type == 'IntConst':
            node_name = ast.value
            num = self.verilog_to_int(node_name)
            if num == 1:
                node_name = "1'b1"
            else:
                node_name = "1'b0"
            # else:
            #     print(num)
            #     assert False
        elif node_type in ['Unot', 'And', 'Or', 'Xor', 'Cond', 'Concat']:
            if node_type == 'Unot':
                node_name = 'INV_X1'
            elif node_type == 'And':
                node_name = 'AND2_X1'
            elif node_type == 'Or':
                node_name = 'OR2_X1'
            elif node_type == 'Xor':
                node_name = 'XOR2_X1'
            elif node_type == 'Cond':
                node_name = 'MUX2_X1'
            elif node_type == 'Concat':
                node_name = 'Concat'
            else:
                print(node_type)
                assert False
        else:
            print(node_type)
            assert False
        return node_name
    
    def unroll_syscall(self, ast):
        nodetype = ast.get_type()
        if(nodetype in ["SystemCall"]):
            ast = ast.args[0]
        return ast

    def verilog_to_int(self, verilog_num):
        num = re.findall(r"(\d+)'(\D)(\d+)", verilog_num)
        if num:
            if 'x' in num[0][2]:
                ret_num = 0
            else:
                if num[0][1] == 'h':
                    ret_num = int(num[0][2], 16)
                elif num[0][1] == 'b':
                    ret_num = int(num[0][2], 2)
                elif num[0][1] == 'd':
                    ret_num = int(num[0][2], 10)
                ## add new cases here
                else:
                    print(verilog_num)
                    assert False
        else:
            ret_num = 0
            # print(verilog_num)
            # assert False

        # print(verilog_num, ret_num)
        # input()
        return ret_num


    def assign(self, ast, parent_name):
        self.rt_flag = 0
        node_type = ast.get_type()
        parent_type = ast.get_parent_type()

        if parent_type == 'Constant':
            node_name = 'Constant' + str(self.const_label)
            self.const_label += 1
            width = self.get_width_num(ast.value)
            val = self.verilog_to_int(ast.value)        
            self.graph.add_decl_node(node_name, parent_type, width,father=None,value=val)
        elif parent_type in ['Operator', 'UnaryOperator']:
            node_name = str(node_type) + str(self.oper_label)
            self.oper_label += 1
            self.graph.add_decl_node(node_name, parent_type)
        elif parent_type in ['Concat', 'Repeat']:
            node_name = str(parent_type) + str(self.oper_label)
            self.oper_label += 1
            self.graph.add_decl_node(node_name, parent_type, 0)
        elif parent_type in ['Identifier', 'Pointer', 'Partselect']: 
            node_name = self.add_new_node(ast)
            self.rt_flag = 1
        elif parent_type == 'SystemCall':
            ast = self.unroll_syscall(ast)
            self.assign(ast, parent_name)
            return
        
        elif parent_type == 'FunctionCall':
            self.func_call(ast, parent_name)
            return
        else:
            print('ERROR, future work')
            print(ast)
            print(node_type)
            print(ast.var)
            assert False
        
        self.graph.add_edge(parent_name, node_name)
        if self.rt_flag == 1:
            return
        for c in ast.children():
            self.assign(c, node_name)
    
    def func_call(self, ast, parent_name):
        node_type = ast.get_type()
        c = list(ast.children())
        func = str(c[0])
        del c[0]
        in_list = []
        for i in c:
            in_list.append(i)
        func_list = self.func_dict[func]
        input_list = func_list[0]
        assign_list = func_list[1]
        sub_dict = dict(zip(input_list, in_list))
        self.add_assign_edge(ast, node_type, sub_dict)

    def get_width_num(self, num):
        is_string = re.findall(r"[a-zA-Z]+\'*[a-z]* |'?'*", num)
        if num in ['0', '1']:
            width = 1    
        elif '\'' in num:
            width = re.findall(r"(\d+)'(\w+)", num)
            width = int(width[0][0])
        elif is_string:
            width = len(num)
        else:
            print('ERROR: New Situation!')
            print(num)
            width = 0
            print(is_string)
            assert False
        
        return width

    
    def eliminate_wires(self, g:Graph):
        print('----- Eliminating Wires in Graph -----')
        for name, node in self.graph.node_dict.items():
            if node.father in self.wire_set:
                # print(name)
                # input()
                self.wire_set.add(name)
        g_node = g.get_all_nodes2()
        interset = g_node & self.wire_set
        ll = len(interset)
        while(len(interset)!=0):
            pre_len = len(interset)
            g = self.eliminate_wire(g)
            g_node = g.get_all_nodes2()
            interset = g_node & self.wire_set
            post_len = len(interset)
            if pre_len == post_len:
                break
        if len(interset) != 0:
            # print('Warning: uneliminated wire: ', len(interset))
            for n in interset.copy():
                neighbor = self.graph.get_neighbors(n)
                if len(neighbor) == 0:
                    self.graph.remove_node(n)
                    interset.remove(n)

            # print('Final uneliminated wire: ', len(interset))
        node_dict = self.graph.node_dict.copy()
        self.graph = g
        self.graph.load_node_dict(node_dict)

    def eliminate_wire(self, g:Graph):
        node_set = g.get_all_nodes()
        for node in node_set:
            node_list = g.get_neighbors(node)
            if node in self.wire_set:
                self.wire_dict[node] = node_list
            else:
                self.temp_dict[node] = node_list
        g_new = Graph()
        for node, node_list in self.temp_dict.items():
            for n in node_list:
                if n in self.wire_dict.keys():
                    wire_assign = self.wire_dict[n]
                    for w in wire_assign:
                        if w:
                            g_new.add_edge(node, w)
                else:
                    g_new.add_edge(node, n)
        return g_new
