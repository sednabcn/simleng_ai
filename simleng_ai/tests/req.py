#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:40:18 2023

@author: delta
"""
import os
from ..resources.io import find_full_path
packlist = ["simleng.py"]
for root, dirs, files in \
        os.walk("/home/delta/Downloads/simleng_ai"):

    packlist_ = [name for name in files
                 if name.endswith('.py') == True
                 if name.startswith('__') == False
                 if name.startswith('.#') == False
                 if name.startswith('#') == False]
    if len(packlist_) > 0:
        packlist.extend(packlist_)
packlist = list(set(packlist))
packlist.sort()



def read_txt_to_list_txt(filename, PATTERNS):
        import re
        output = []
        with open(filename, 'r', encoding='utf-8') as f:
            for n, line in enumerate(f):
                l1 = line.split('\n')
                for pattern in PATTERNS:
                    if re.search(pattern, line) \
                        and line.startswith(pattern) == True:

                            output.append(l1[0])
        return output


def all(list_txt,PATTERNS):
    import re
    __all__ = []
    rx = re.compile(r'([^from]\w+)')
    for line in list_txt:
        rs=rx.findall(line)
        rss=[x.strip().lstrip('.').lstrip(',') for x in rs] 
        rsq=[x for x in rss  if x.strip() not in PATTERNS]
        nn=rss.index('import')
        __all__.extend(rss[:nn])
    return list(set(__all__))

output_list=[]
PATTERNS=['from', 'import']
PATH="/home/delta/Downloads/simleng_ai"
for file in packlist:
    filename=find_full_path(file, PATH)
    output=read_txt_to_list_txt(filename, PATTERNS)
    if isinstance(output, list) == True and len(output) > 0:
               output_list.extend(output)
xx=all(output_list,PATTERNS)

#=======================================
print(packlist)

output_list

xx
