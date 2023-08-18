# ==["read_txt_to_dict_line","read_txt_to_dict_key_multiple","read_txt_to_dict","read_txt_to_list_txt"]

from ..resources.sets import x_add_list


def read_txt_to_dict_line(filename, PATTERNS, LIST_LINE_PAR):
    import re, linecache

    """
        Return a dictionary to read a txt file
        Entries:
        filename: txt file
        PATTERNS: List of pattern
        LIST_LINE_PAR with two parameters
        Output:
        Dict[PATTERNS] or Dict[n] in case of there isn't patterns
        """
    output = {}
    # Read line by line to get the MACROS and the information based on it
    # try:
    with open(filename) as input_file:
        for n, line in enumerate(input_file):
            # case 1: PATTERNS=[]
            if len(PATTERNS) == 0:
                output[n] = line.strip.split()
            # case 2 : PATTERNS not empty
            nm = x_add_list(LIST_LINE_PAR, n)
            for pattern in PATTERNS:
                for match in re.finditer(r"\b" + pattern + r"\b", line):
                    u1 = linecache.getline(filename, nm[0], module_globals=None).split()
                    v1 = linecache.getline(filename, nm[1], module_globals=None).split()
                    if len(v1) > 0:
                        output[match.group()] = [{u: v} for (u, v) in zip(u1, v1)]
                    else:
                        output[match.group()] = [{u: ""} for u in u1]

    # except:
    #     # command line param and is a file
    #   raise IndexError("There must be a filename specified.")

    return output


def read_txt_to_dict_key_multiple(filename, PATTERNS, key_MULTIPLE):
    """It has not been tested yet"""
    output = {}
    lmultiple = []
    with open(filename, "r") as f:
        for n, line in enumerate(f):
            l1 = line.split()
            if any([item for item in l1 if item in PATTERNS]):
                key = l1[0]
                nn = n
                output[key] = []
                # print(n,key)
            elif len(l1) > 0 and (n - nn > 0):
                if (n - nn) == 1:
                    l11 = l1
                else:
                    l12 = l1
                    if key == key_MULTIPLE:
                        lmultiple.extend(l12)
                        # print('l122:',l122,l12)
                        output[key] = [{l11[0]: l122[:]}]
                        # print(MACROS[key])
                        # print("===========")
                    if len(l12) != len(l11):
                        pp = len(l11) - len(l12)
                        l12.extend(pp * ["-"])
                    if key != key_MULTIPLE:
                        output[key] = [{u: v} for (u, v) in zip(l11, l12)]

                if nn == n:
                    break
            else:
                pass
    return output


def read_txt_to_dict(filename, PATTERNS):
    from ..resources.sets import list_to_list_of_dict

    output = {}
    with open(filename, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            l1 = line.split()
            if any([item for item in l1 if item in PATTERNS and l1[0][0] != "#"]):
                key = l1[0]
                nn = n
                output[key] = []
                l122 = []
            elif len(l1) > 0 and (n - nn == 1) and l1[0] != "#":
                l11 = l1
                l122 = [l11]
            elif len(l1) > 0 and (n - nn > 1) and l1[0] != "#":
                l12 = l1
                if len(l12) != len(l11):
                    pp = len(l11) - len(l12)
                    l12.extend(pp * ["-"])
                l122.append(l12)
            else:
                output.update({key: list_to_list_of_dict(l122)})
        output[key] = list_to_list_of_dict(l122)
    return output


def read_txt_to_list_txt(filename, PATTERNS):
    import re

    output = []
    with open(filename, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            l1 = line.split("\n")
            for pattern in PATTERNS:
                if re.search(pattern, line) and line.startswith(pattern) == True:
                    output.append(l1[0])
    return list(set(output))
