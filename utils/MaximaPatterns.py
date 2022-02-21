'''
From the set of closed frequent patterns, the class MaximalPatterns extracts maximal patterns.
It takes as input a text file with the list of patterns as strings.
Only the maximal subset is returned.
'''

from itertools import combinations, product
import re

class MaximalPatterns:

    def __init__(self, patterns_file, output_file):
        self.patterns = patterns_file
        self.output_file = output_file


    def execute(self):
        patterns = self.read_files()
        pstr = [''.join(p) for p in patterns]
        pstr.sort(key=len, reverse=True)
        not_max=[]
        for p, k in combinations(pstr, 2):
            if p.find(k) != -1:
                not_max.append(k)
        output = [re.findall('[-+]?\d', p) for p in pstr if p not in not_max]
        self.write_patterns_to_file(output)


    def read_files(self):
        p = open(self.patterns, 'r')
        p = p.readlines()
        pat = self.parse_patterns(p)
        return pat


    def parse_patterns(self, p):
        patterns = []
        for el in p:
            out = el[:el.find (']') + 1]
            out = out.replace ('[', '').replace(']', '').replace("'", '').replace(',', ' ')
            out = out.split()
            patterns.append(out)
        return patterns



    def write_patterns_to_file(self, patterns):
        file_ = open(self.output_file, 'a')
        for p in range(0, len(patterns)):
            file_.write(str(patterns[p]) + "\n")
        file_.close()