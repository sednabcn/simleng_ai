import os
import sys
from ..resources.io import find_full_path
from distutils.dir_util import copy_tree


class MyDB(object):
    """
    db-new database
    store-place to store one
    kind:-structured(table),-non-structured
    path=db_client:path of the new database
    
    """
    def __init__(self, db, store, kind=None, path=None):
        self.db = db
        self.db_client = path
        self.db_kind = kind
        self.db_store = find_full_path(store)
        if path == None:
            self.db_client = find_full_path(self.db)

        #print(self.db)
        #print(store)
        #print(self.db_store)
        #print(self.db_client)
        #print(input("PAUSE"))
        
    def datasets_store(self):
        _, dirs_store, files_store = list(os.walk(self.db_store))[0]
        _, dirs_client, files_client = list(os.walk(self.db_client))[0]

        if self.db_kind == "table":
            # if self.db_kind=='table':
            if self.db not in dirs_store:
                self.db_client_in_store = os.path.join(self.db_store, self.db)
                os.mkdir(self.db_client_in_store)
                return copy_tree(self.db_client, self.db_client_in_store)
        else:
            pass

    
"""
EXAMPLE

python 3.8.10 (default, May 26 2023, 14:05:08) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from ..resources.db import MyDB
>>> MyDB('pimas','datasets',kind='table').datasets_store()
['/home/delta/Downloads/simleng_ai/datasets/pimas/columns.txt~', '/home/delta/Downloads/simleng_ai/datasets/pimas/columns.txt', '/home/delta/Downloads/simleng_ai/datasets/pimas/pima-indians-diabetes.csv~', '/home/delta/Downloads/simleng_ai/datasets/pimas/#columns.txt#', '/home/delta/Downloads/simleng_ai/datasets/pimas/pima-indians-diabetes.csv', '/home/delta/Downloads/simleng_ai/datasets/pimas/pima-indians-diabetes0.csv', '/home/delta/Downloads/simleng_ai/datasets/pimas/pima-indians-diabetes1.csv', '/home/delta/Downloads/simleng_ai/datasets/pimas/pimas-indias-diabetes-bak.csv']
"""
