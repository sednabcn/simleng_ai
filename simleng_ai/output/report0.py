#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:10:40 2023

@author: agagora

This programm creates a Report of Simleng Project
===================================================
python report.py input_file output_file
===================================================
"""

import os
import sys
import pandas as pd
from simleng_ai.ini.init import Init
from simleng_ai.output.table import Table_results
from simleng_ai.resources.output import images_to_output_results 
from simleng_ai.resources.io import input_to_dict_table
from tabulate import tabulate

top_level_path='/home/agaora/Downloads/Aug27-1623'

# args to script
#input_entry = sys.argv[1]
#output_entry = sys.argv[2]
#input_entry= '/home/agagora/Downloads/Aug27-1623/PROJECTOML/simleng_ai/simleng_ai/input_file/simlengin2708.txt'
input_entry='simlengin2708.txt'
output_entry='/home/agagora/Downloads/Aug27-1623/PROJECTOML/simleng_ai/simleng_ai/output/pimas/pimas_result.txt'
top_level_path='/home/agagora/Downloads/Aug27-1623/PROJECTOML/simleng_ai/simleng_ai'
# input basic information 
#WORKDIR = input("Enter the workdir (exec simleng_ai) :")

WORKDIR='/home/agagora'

#page_title_text = input("Enter a Project title :")

page_title_text='Data Project'

#title_text = input("Enter a Report title :")

title_text='Pimas Report'

#text = input("Enter a short description of the Report :")

text="Classification"

# get information from input_entry
MACROSIN={}
MACROSIN = Init(input_entry).get_macros()

data=MACROSIN['DATA_PROJECT'][0]
if isinstance(data,dict):
    dataset=data['DATASET']

data_source=MACROSIN['DATA_PROJECT'][4] 
if isinstance(data_source,dict):
    source =data_source["DATASOURCE"]

data_goal =MACROSIN['TARGET'][0]
if isinstance(data_goal,dict):
    goal = data_goal["GOAL"]

# show MICROSIN TABLE    
input_to_table = input_to_dict_table(dict(MACROSIN.items()),idoc=0)

# detect and moving figures from the WORKDIR to open_results_dir
images_to_output_results(dataset,WORKDIR,'output_results','png',top_level_path=top_level_path)
# get information from output_entry
filename=str(page_title_text)+'_'+ str(dataset) + '.'+'txt'

# create report page
with open(output_entry,'r') as output: 
     with open(filename,'w') as report:
        # title of report
        report.write("Report:",page_title_text)
        # title of text
        report.write("Title of text:",title_text)
        # goal
        report.write("Goal:",goal)
        # text: description
        report.write("Description:",text)

        for line in output:
            report.write(line)
        output.close()
        report.close()
            
"""
with open('TEST.txt', 'r') as firstfile,
    open('RESULT.html', 'a') as secondfile:
       # read content from first file
       for line in firstfile:
       # append content to second file
           secondfile.write(line + '<br/>'
"""                     
                   
print(tabulate(input_to_table, headers=input_to_table.keys(), tablefmt="html"))


"""
#=====================================================================
html_text = '<html>\n<body>\n'

for line in text.split('\n'):
    html_text += f'<p>{line}</p>\n'

html_text += '</body>\n</html>'

with open('example.html', 'w') as file:
    file.write(html_text)
"""
