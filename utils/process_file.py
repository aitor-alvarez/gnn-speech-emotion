def create_dictionary(path):
    patterns = read_directory(path)
    dictionary =[]
    vocab = [dictionary.append(p) for p in patterns if p not in dictionary]
    return dictionary


def read_file(patterns):
    p = open(patterns, "r")
    p = p.readlines()
    pat = parse_patterns(p)
    return pat

def parse_patterns(p):
    patterns = []
    for el in p:
        out = el[:el.find(']') + 1]
        out = out.replace('[', '').replace(']', '').replace("'", '').replace(',', ' ')
        out = out.split()
        patterns.append(out)
    return patterns


def read_directory(path_file):
    all_patterns = read_file(path_file)
    return all_patterns