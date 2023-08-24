# ==["table_from_dict","table","color_text","image_to_report"]


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
def image_to_report(idoc,imname,format):
    """Putting images in the report"""
    import matplotlib.pyplot as plt
    
    if idoc>=1:
        return plt.savefig(imname+'.'+str(format))
    elif idoc==0 or idoc==None:
        return plt.show()
    else:
        pass
