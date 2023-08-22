# ==["file_reader","file_writer","find_full_path"]


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
    from os.path import isfile, isdir

    rt = []
    if path == None:
        path = os.path.expanduser("~")

    for root, dirs, files in os.walk(path):
        if name in files or name in root:
            rt.append(root)

    if len(rt) > 0:
        rt.sort(key=len)
        new_path = rt[0]
        full_path = os.path.join(new_path, name)
    else:
        return print("%s path has not been found" % (name))

    if isfile(full_path):
        return full_path
    else:
        return new_path
