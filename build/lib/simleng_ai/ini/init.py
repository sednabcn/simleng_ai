#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:22:49 2023

@author: delta
"""

#read the input file

import sys
from resources.scrapping import read_txt_to_dict
from resources.io import find_full_path

#filename=sys.argv[1]
#file_input='simlengin.txt'

MACROS=['Simlengin.txt','DATA_PROJECT','FILES','LISTFILES','FEATURES','TARGET','STRATEGIES','PARAMS']


class Init:

     def __init__(self,file_input,score=None,NEWMACROS=None):
           self.file_input=find_full_path(file_input)
           self.score=score
           self.NEWMACROS=NEWMACROS
           

     def get_macros(self):     
         return read_txt_to_dict(self.file_input,MACROS)

     def update_macros(self,MACROS,NEWMACROS):
         self.NEWMACROS=NEWMACROS
         # to update parameters from the input_file given by Simleng
         # you have to open a file for writting
         return read_txt_to_dict(self.file_input,MACROS+self.NEWMACROS)

#MACROSIN = Init(file_input,score).get_macros()
#print(dict(MACROSIN.items()))
     
#if __name__ =='__main__':
#     #MACROSIN = Init.get_macros(sys.argv[1])
#     MACROSIN = Init.get_macros(filename)
#     print(dict(MACROSIN.items()))
#     #main(sys.argv[1:])
"""
EXAMPLE Scrapping simlengin.txt
Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from init import Init
/home/agagora/Downloads/simlengOLD/simleng/simlengin.txt
{'Simlengin.txt': [], 'DATA_PROJECT': [{'DATASET': 'pimas'}, {'TYPE': 'numeric'}, {'STRUCT': 'table'}, {'SYNTHETIC': 'False'}, {'DATASOURCE': 'table'}, {'IMBALANCE': 'False'}, {'DUMMY': 'True'}], 'FILES': [{'NUMBER': '1'}, {'FORMAT': 'csv'}, {'reader': 'pandas'}, {'mode': 'read'}, {'index_col': '0'}, {'header': 'None'}, {'sep': '\\n'}], 'LISTFILES': [{'filename': 'pima-indians-diabetes.csv'}], 'TARGET': [{'GOAL': 'CLASSIFICATION'}, {'NCLASS': '2'}, {'METHOD': 'SUPERV'}, {'SOLVER': 'stats'}, {'EXPECTEDSCORE': '0.20'}, {'SPLITDATA': '-'}]}

EXAMPLE Scrapping simlengin.txt with unicode='UTF-8'

Python 3.11.2 (main, Mar 13 2023, 12:18:29) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from init import Init
/home/agagora/Downloads/simleng_ai/input_file/simlengin.txt
{'Simlengin.txt': [], 'DATA_PROJECT': [{'DATASET': 'pimas'}, {'TYPE': 'numeric'}, {'STRUCT': 'table'}, {'SYNTHETIC': 'False'}, {'DATASOURCE': 'table'}, {'IMBALANCE': 'False'}, {'DUMMY': 'True'}], 'FILES': [{'NUMBER': '1'}, {'FORMAT': 'csv'}, {'reader': 'pandas'}, {'mode': 'read'}, {'index_col': '0'}, {'header': 'None'}, {'sep': '\\n'}], 'LISTFILES': [{'filename': 'pima-indians-diabetes.csv'}], 'TARGET': [{'GOAL': 'CLASSI'}, {'NCLASS': '2'}, {'METHOD': 'SUPERV'}, {'SOLVER': 'stats'}, {'SCORE': '---'}, {'SPLITDATA': '0.25'}]}
"""
