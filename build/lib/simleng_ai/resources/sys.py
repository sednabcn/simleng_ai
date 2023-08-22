# =[get_class_for_name]


def get_class_for_name(module_name, class_name):
    import importlib

    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c
