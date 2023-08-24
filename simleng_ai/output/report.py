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
input_entry=sys.argv[1]
output_entry=sys.argv[2]

WORKDIR=input("Enter the workdir (exec simleng_ai)")
page_title_text=input("Enter a Project title:")
title_text=input("Enter a Report title")
text=input("Enter a short description of the Report")

MACROSIN=Init(input_entry)

input_to_table=pd.DataFrame.from_dict(MACROSIN)

Table_results(0,input_to_table, ".3f", "fancy_grid",title_text, 60
        ).print_table()
