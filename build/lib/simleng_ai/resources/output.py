# ==["table_from_dict","table","color_text","image_to_report","images_to_output_results"]


def table_from_dict(x, headers, floatfmt, style, title, width):
    """Print Data Table with tabulate..."""
    from tabulate import tabulate

    """
            | x--Data Table
            | floatfmt--float format number
            | style="grid","simple","fancy_grid","Latex","html"
            | from tabulate import tabulate
            """
    return (
        print("\n"),
        print(
            title.center(width),
            tabulate(
                x, headers=headers, tablefmt=style, stralign="center", floatfmt=floatfmt
            ),
            sep="\n",
        ),
        print("\n\n"),
    )


def color_text(x, facecolor):
    from termcolor import colored

    return colored(x, facecolor)


def table(x, floatfmt, style, title, width):
    """Print Data Table with tabulate..."""
    from tabulate import tabulate

    """
            | x--Data Table
            | floatfmt--float format number
            | style="grid","simple","fancy_grid","Latex","html"
            | from tabulate import tabulate
            """
    return (
        print("\n"),
        print(
            title.center(width),
            tabulate(
                x, headers="keys", tablefmt=style, stralign="center", floatfmt=floatfmt
            ),
            sep="\n",
        ),
        print("\n\n"),
    )


def image_to_report(idoc, imname, imformat):
    """Putting images in the report"""
    import matplotlib.pyplot as plt

    if idoc >= 1:
        return plt.savefig(imname + "." + str(imformat))
    elif idoc == 0 or idoc == None:
        return plt.show()
    else:
        pass

def images_to_output_results(dataset,workdirpath,store,imformat,top_level_path=None):
    """movement images from workdir to store
    dataset: name of dataset
    workdirpath: location of images
    store: place to keep the images("output_dir")
    imformat:str or list of str images formats
    """
    import os
    from ..resources.io import find_full_path
    import shutil
    
    # check whether imformat is str or list
    if isinstance(imformat,str):
         size=1
         imf=[imformat]
    elif isiinstance(imformat,list):
        size=len(imformat)
        imf=imformat.copy()
    else:
        pass

    # Level of working
    if top_level_path==None:
        path_store=find_full_path(store)
    else:
        path_store=find_full_path(store,top_level_path)
    
    # check if dataset dir exist in store    
    if dataset not in store:
        path_dataset_in_store=os.path.join(dataset,path_store)
        os.mkdir(path_dataset_in_store)
    else:
        pass
    # copy files from a dir to another
    # check whether the files endswith(imformat) are in workdir
    listimages=[files for files in os.listdir(workdirpath) for fm in imf if files.endswith(fm)]
    # shutil.copy(src,dst)
    for files in listimages:
        path_file=os.path.join(files,workdirpath)
        shutil.copy(path_file,path_dataset_in_store)
    return [f for f in os.listdir(path_dataset_in_store) for fm in imf if f.endswith(fm)]
