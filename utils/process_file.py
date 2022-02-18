import os


def create_dictionary(path):
    patterns = read_directory(path)
    songs = process_patterns(patterns)
    dictionary =[]
    vocab = [dictionary.append(p) for d in songs for p in d if p not in dictionary]
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
        patterns.append((out, int(el[el.find(']') + 1:].replace('\n',''))))
    return patterns


def read_directory(path):
    all_patterns = [read_file(path+f) for f in os.listdir(path) if f.endswith('_maximal.txt')]
    return all_patterns


def process_patterns(pat):
    docs=[]
    for doc in pat:
        docu=[]
        for d in doc:
            #docu.append('_'.join(d[0]))
            docu.append(d[0])
        docs.append(docu)
    return docs
