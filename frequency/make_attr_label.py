import json
import collections

with open('dscrp_selected.txt','r') as f:
    lines = f.readlines()

idx = 0
attr_dict = collections.defaultdict(lambda: 0)
for line in lines:
    line = line.strip().split(':')
    if line[-1] == '':
        key_out = line[0]
    else:
        key_in = line[0].split()
        for k in key_in:
            key = '{}_{}'.format(key_out,k)
            if key not in attr_dict:
                attr_dict[key] = idx
                idx += 1 

json.dump(attr_dict,open('attribute.json','w'))