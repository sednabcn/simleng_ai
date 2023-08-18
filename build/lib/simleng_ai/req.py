#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:40:18 2023

@author: delta
"""

package = "simleng_ai"

def build_package_list(package):
    import os
    from resources.io import find_full_path
    path=find_full_path(package)
    packlist=[]
    for root, dirs, files in os.walk(path):
           
            packlist_ = [name for name in files
                 if name.endswith('.py') == True
                 if name.startswith('__') == False
                 if name.startswith('.#') == False
                 if name.startswith('#') == False]
            if len(packlist_) > 0:
                packlist.extend(packlist_)
    packlist = list(set(packlist))
    packlist.sort()
    return packlist


def read_txt_to_list_txt(filename, PATTERNS):
        import re
        output = []
        with open(filename, 'r', encoding='utf-8') as f:
            for n, line in enumerate(f):
                l1 = line.split('\n')
                
                for pattern in PATTERNS:
                    if re.search(pattern, line)\
                        and line.startswith(pattern) == True:
                            output.append(l1[0])
                  
        return list(set(output))


def all_names_import(list_txt,PATTERNS=None):
    import re
    __all__ = []
    rx = re.compile(r'([^from]\w+)')
    for line in list_txt:
        rs=rx.findall(line)
        rss=[x.strip().lstrip('.').lstrip(',') for x in rs] 
        #rsq=[x for x in rss  if x.strip() not in PATTERNS]
        nn=rss.index('import')
        __all__.extend(rss[:nn])
    return list(set(__all__))

def all_names_class(list_txt,PATTERNS=None):
    import re
    __all__=[]
    rx=re.compile(r'(^class\s(\w+))')
    for line in list_txt:
        if rx.search(line):
            __all__.append(rx.search(line).group(2))
    
    all_classes=list(set(__all__))
    all_classes.sort()
    return all_classes

#PATTERNS=['from', 'import']
#PATH="/home/delta/Downloads/simleng_ai"

def requirements(package,PATTERNS):
    "Get the packages requirements to import"
    from resources.io import find_full_path
    from resources.scrapping import read_txt_to_list_txt
    from resources.package import build_package_list
    
    
    path=find_full_path(package)
    
    packlist=build_package_list(package)
    
    output_list=[]    
    
    for file in packlist:
        
        filename=find_full_path(file, path)
    
        output=read_txt_to_list_txt(filename, PATTERNS)
        
        if isinstance(output,list)==True and len(output)>0:
            output_list.extend(output)
            
    return list(set(output_list))

print("===================1=======================")
packlist=build_package_list(package)
print("\n".join(packlist))
print("====================2=======================")
PATTERNS=['class']
output_list=requirements(package,PATTERNS)

print("\n".join(output_list))
print("=====================4========================")
if 'import' in PATTERNS:
    xx=all_names_import(output_list)
    print("\n".join(xx))
print("=================================================")
if 'class' in PATTERNS:
   yy=all_names_class(output_list)
   print("\n".join('"{0}"'.format(w) for w in yy))    
    