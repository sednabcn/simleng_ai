#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:10:40 2023

@author: agagora

This programm creates a Report of Simleng Project
===================================================
python report.py input_file output_file SOURCEDIR
===================================================
"""

import os
import sys
import pandas as pd
from simleng_ai.ini.init import Init
from simleng_ai.output.table import Table_results
from simleng_ai.resources.output import mv_files_to_dir_in_dir 
from simleng_ai.resources.io import input_to_dict_table,find_full_path
from tabulate import tabulate


# ALERT: IT IS NECCESARY A CONTROL HERE
# args to script
input_entry = sys.argv[1]
output_entry = sys.argv[2]
input_entry_path=find_full_path(input_entry)
output_entry_path=find_full_path(output_entry)
source_dir_path=sys.argv[3]
SOURCEDIR=source_dir_path.split('/')[-1]
# input basic information 

WORKDIR = os.getcwd()
if not output_entry_path:
    output_entry_path=os.path.join(WORKDIR,output_entry)

if os.path.isfile('header.txt'):
    with open('header.txt','r') as hd:
        lines=hd.readlines()
        SOURCEDIR=str(lines[0])
        page_title=lines[1]
        report_title=lines[2]
        goal_text=lines[3]
        description_text=lines[4]
        hd.close()
        
else:
    with open ('header.txt','w') as wd:
          #SOURCEDIR=input("Enter a src dir:")
          wd.write(SOURCEDIR)
          wd.write('\n')
          page_title= input("Enter a Project title :")
          wd.write(page_title)
          wd.write('\n')
          report_title = input("Enter a Report title :")
          wd.write(report_title)
          wd.write('\n')
          goal_text=input("Enter a main goal of the project :")
          wd.write(goal_text)
          wd.write('\n')
          description_text = input("Enter a short description of the Report :")
          wd.write(description_text)
          wd.write('\n')
          wd.close()

#source_dir_path=find_full_path(SOURCEDIR)

top_level_path=source_dir_path

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


def image_to_html(pathtohtml,image):
   """display image in html""" 
   str='<img src="text"  alt=" ",style="width:100%">'
   with open(pathtohtml,'w') as t:
      t.write('<!DOCTYPE html>' + '\n')
      t.write('<html>' +'\n')
      t.write('<figure>' +'\n')
      t.write(str.replace("text",image)+'\n')
      t.write('</figure>' + '\n')
      t.write('</hml>' + '\n')
      t.close()

def getmtime(x):
    import os
    return os.path.getmtime(x)


def read_image_to_html(htmlfile):
   with open(htmlfile,'r',encoding='utf-8') as f:
        print(f.read())
        f.close()

png_files=[]
for files in os.listdir(WORKDIR):
    if files.endswith('png'):
        png_files.append(files)

png_sort_files=sorted(png_files,key=getmtime)

for i in range(len(png_files)):
    image=png_sort_files[i]
    path_to_html=WORKDIR+'/'+dataset+'_'+ str(i) +'.html'      
    image_to_html(path_to_html,image)
    read_image_to_html(path_to_html)

# get information from output_entry
filename=str(page_title) +'_'+ str(dataset)
file_txt=filename + '.'+'txt'
file_html=filename+ '.'+'html'
i=0
str='<img src="text"  alt=" ",style="width:100%">'
# create report page
with open(output_entry_path,'r') as output:
  with open(input_entry_path,'r') as input:
     with open(file_txt,'w') as report:
        # title of project
        report.write(page_title)
        report.write('\n\n\n')
        # title of report
        report.write(report_title)
        report.write('\n\n\n')
        # goal
        report.write(goal_text)
        report.write('\n\n\n')
        # text: description
        report.write(description_text)
        report.write('\n\n\n')
        # input_file
        report.write("DATA INPUT FILE")
        report.write('\n\n\n')
        for line in input:
            report.write(line)
        report.write('\n\n\n')
        for line in output:
            """
            if line.startswith('Fig'):
                 report.write("\n\n")
                 report.write('<!DOCTYPE html>' + '\n')
                 report.write('<html>' +'\n')
                 report.write('<figure>' +'\n')
                 report.write(str.replace("text",png_sort_files[i])+'\n')
                 report.write('</figure>' + '\n')
                 report.write('</hml>' + '\n')
                 report.write("\n\n")
                 i+=1
            if "Fig" not in line:
            """
            report.write(line)
        output.close()
        input.close()
        report.close()
           
#=========================================================================
# Movement of files to ./ouput_results/dataset directory

# files
# detect and moving figures from the WORKDIR to open_results_dir
for formatt in ['png','txt','html']:
    mv_files_to_dir_in_dir(dataset,WORKDIR,'output_results',formatt,top_level_path=top_level_path)

#====================================================================================
