from __future__ import absolute_import
from __future__ import print_function
import sys
import os, time
from optparse import OptionParser

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyverilog
from pyverilog.vparser.parser import parse
from AST_analyzer import *


def main(clk=None, reset=None, design_name=None, cmd=None):
    start_time = time.perf_counter()
    INFO = "Verilog code parser"
    VERSION = pyverilog.__version__
    USAGE = "Usage: python example_parser.py file ..."

    def showVersion():
        print(INFO)
        print(VERSION)
        print(USAGE)
        sys.exit()

    optparser = OptionParser()
    optparser.add_option("-v", "--version", action="store_true", dest="showversion",
                         default=False, help="Show the version")
    optparser.add_option("-I", "--include", dest="include", action="append",
                         default=[], help="Include path")
    optparser.add_option("-D", dest="define", action="append",
                         default=[], help="Macro Definition")
    optparser.add_option("-K", dest="clk", action="append",
                         default=[], help="clock name")
    optparser.add_option("-R", dest="reset", action="append",
                         default=[], help="reset name")
    optparser.add_option("-N", dest="Name", action="append",
                         default=[], help="Design Name")
    optparser.add_option("-C", dest="cmd", action="append",
                         default=[], help="Design command")
    optparser.add_option("-T", dest="top", action="append",
                         default=[], help="Top Name")
    (options, args) = optparser.parse_args()

    if options.Name:
        design_name = options.Name[0]
    if options.cmd:
        cmd = options.cmd[0]
    if options.clk:
        clk_name = options.clk[0]
    if options.reset:
        reset_name = options.reset[0]
    if options.top:
        top_name = options.top[0]

    

    filelist = args
    if options.showversion:
        showVersion()

    for f in filelist:
        if not os.path.exists(f):
            raise IOError("file not found: " + f)

    if len(filelist) == 0:
        showVersion()
    

    ast, directives = parse(filelist,
                            preprocess_include=options.include,
                            preprocess_define=options.define)

    print('Verilog2AST Finish!')
    # ast.show()


    ast_analysis = AST_analyzer(ast, clk_name, reset_name)
    
    ast_analysis.AST2Graph(ast, filelist[0])

    file_lines = ast_analysis.line_lst

    with open (f'/data/usr/vlg2netlist/generated_netlist_file/{top_name}_{design_name}_TYP.syn.v', 'w') as f:
        for line in file_lines:
            f.writelines(line)
    
    with open (f'/data/usr/vlg2netlist/generated_name_map/{design_name}_name_map.json', 'w') as f:
        json.dump(ast_analysis.name_map, f)
    with open (f'/data/usr/vlg2netlist/generated_name_map/{design_name}_name_map2.json', 'w') as f:
        json.dump(ast_analysis.name_map2, f)

    os.system(f'cp ./sdc_template.sdc /data/usr/vlg2netlist/generated_sdc_file/{top_name}_{design_name}_TYP.sdc')

    

if __name__ == '__main__':
    main()
