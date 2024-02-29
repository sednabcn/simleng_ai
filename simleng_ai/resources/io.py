# ==["file_reader","file_writer","find_full_path","input_to_dict_table","checking_input_list"]


def file_reader(file, type_file, *pars):
    import pandas as pd
    import csvreader

    # import ZipFile
    type_file_pd = [
        "csv",
        "xls",
        "xlsx",
        "json",
        "sql",
        "html",
        "pickle",
        "tsv",
        "excel",
    ]
    type_file_reader = ["tar", "tar.gz", "pdf", "HDF5", "docx", "mp3", "mp4"]
    type_file_open = ["txt"]
    if type_file in type_file_pd:
        if type_file == "csv":
            return pd.read_csv(file)
        elif type_file == "xls":
            return pd.read_excel(file)
        elif type_file == "xlsx":
            return pd.read_excel(file)
        elif type_file == "json":
            return pd.read_json(file)
        elif type_file == "sql":
            return pd.read_sql(file)
        elif type_file == "html":
            return pd.read_html(file)
        elif type_file == "pickle":
            return pd.read_pickle(file)
        elif type_file == "tsv":
            pass
        elif type_file == "excel":
            return pd.read_excel(file)
        else:
            pass
    elif type_file in type_file_open:
        pass
    elif type_file in type_file_reader:
        pass
    else:
        pass


def file_writer(file, type_file, *pars):
    import pandas as pd
    import csvwriter

    # import ZipFile
    type_file_pd = [
        "csv",
        "xls",
        "xlsx",
        "json",
        "sql",
        "html",
        "pickle",
        "tsv",
        "excel",
    ]
    type_file_reader = ["tar", "tar.gz", "pdf", "HDF5", "docx", "mp3", "mp4"]
    type_file_open = ["txt"]
    if type_file in type_file_pd:
        if type_file == "csv":
            return pd.read_csv(file)
        elif type_file == "xls":
            return pd.read_excel(file)
        elif type_file == "xlsx":
            return pd.read_excel(file)
        elif type_file == "json":
            return pd.read_json(file)
        elif type_file == "sql":
            return pd.read_sql(file)
        elif type_file == "html":
            return pd.read_html(file)
        elif type_file == "pickle":
            return pd.read_pickle(file)
        elif type_file == "tsv":
            pass
        elif type_file == "excel":
            return pd.read_excel(file)
        else:
            pass
    elif type_file in type_file_open:
        pass
    elif type_file in type_file_reader:
        pass
    else:
        pass


def find_full_path(name, path=None):
    import os
    import re
    from os.path import isfile, isdir
    #Becarefull not getting well directories location
    rt=[]
    if path == None:
        path = os.path.expanduser("~")
    
    if re.match(r'\w+\.\w+',name):
       
       for root, dirs, files in os.walk(path):
        if name in root or name in files:
            rt.append(root)
        elif name in dirs:
             rt.append(root)
        else:
            pass
    else:
        for root,dirs,files in os.walk(path):
            if str(name) in os.listdir(root):
                rt.append(os.path.join(root,name))
            elif str(name) in dirs:
                rt.append(os.path.join(root,dirs,dir))
            else:
                for items in files:
                    if items.startswith(name):
                        rt.append(root)
                    else:
                        pass
            
    if len(rt) > 0:
        rt.sort(key=len)
        new_path = rt[0]
        full_path = os.path.join(new_path, name)
    else:
        return print("%s path has not been found" % (name))

    if isfile(full_path) or isdir(full_path):
        return full_path
    else:
        return new_path

def input_to_dict_table(dict0,idoc=None):
    """convert input template to dataframe.from_dict"""
    import pandas as pd
    from tabulate import tabulate
 
    dict1={}
    for i in dict0.keys():
        dict1[i]={}
        for j in dict0[i]:
            dict1[i].update(j)
    
    df=pd.DataFrame.from_dict({(i,j):dict1[i][j] for i in dict1.keys() for j in dict1[i].keys()},orient='index')
            
    if idoc>=1:
        tabulate(df,tablefmt='html')
    else:
        print(tabulate(df,tablefmt='grid'))
    return df

def checking_input_list(item,list0):
    """Checking an item is a member of a list"""
    if item not in list0:
        list0.append(item)
        print("Checking the input_file, please")
    else:
        pass
