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
from ..ini.init import Init
from ..output.table import Table_results
from ..resources.db import MyDB

# args to script
input_entry = sys.argv[1]
output_entry = sys.argv[2]

# input basic information 
WORKDIR = input("Enter the workdir (exec simleng_ai)")
page_title_text = input("Enter a Project title:")
title_text = input("Enter a Report title")
text = input("Enter a short description of the Report")

# get information from input_entry
MACROSIN = Init(input_entry)

data=MICROSIN['DATA_PROJECT'][0]
if isinstance(data,dict):
    dataset=data['DATASET']

data_source=MICROSIN['DATA_PROJECT'][4] 
if isinstance(data_source,dict):
    source =data_source["DATASOURCE"]
    
MyDB(dataset, "output_results", kind=source).datasets_store()    
input_to_table = pd.DataFrame.from_dict(MACROSIN)
Table_results(0, input_to_table, ".3f", "fancy_grid", title_text, 60).print_table()

# detect and moving figures from the WORKDIR to open_results_dir

# get information from output_entry
