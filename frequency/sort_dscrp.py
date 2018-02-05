import collections
import re
import json

def load_dscrp(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()

    # d = {}
    # for line in lines:
    #     line = line.strip().split(':'.encode('utf-8'))
    #     if line[-1] != '':
    #         d[line[0]] = int(line[-1])

    # DictList= sorted(d.items(), key=lambda x: x[1], reverse=True)
    # with open('sorted_dscrp.txt','w') as f:
    #     for i in DictList:
    #         f.write('{}:{} \n'.format(i[0],i[1]))

    d = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for line in lines[:-1]:
        line = line.strip().split(':'.encode('utf-8'))
        if line[-1] == '':
            key = line[0]
            continue
        d[key][line[0]] = int(line[-1])
    return d

def sorted_dscrp_key():
    d = load_dscrp('dscrp_furniture.txt')
    keys = collections.defaultdict(lambda: 0)
    for key in d.keys():
        for val in d[key].values():
            keys[key] += val
    dictList = sorted(keys.items(), key=lambda x: x[1], reverse=True)

    with open('sorted_dscrp_key.txt','w') as f:
        for item in dictList:
            f.write(item[0]+':\n')
            for i in d[item[0]].items():
                f.write('{}:{}\n'.format(i[0],i[1]))

def make_label():
    d = load_dscrp('dscrp_selected.txt')
    attr = {}
    for key in d.keys():
        for val in d[key].keys():
            val = val.decode('utf-8')
            val = val.replace(u'/',u' ')
            val = val.strip(' ')
            val = val.split()
            for _ in val:
                attr[u'{}_{}'.format(key.decode('utf-8'),_)] = 1

    attr_dump = {}
    for idx,key in enumerate(attr.keys()):
        attr_dump[key] = idx

    json.dump(attr_dump, \
        open('/core1/data/home/shizhan/jiahuan/furniture/label/label/label_v1/attribute.json','w'))
    with open('/core1/data/home/shizhan/jiahuan/furniture/label/label/label_v1/attribute.txt','w') as f:
        for key,val in attr_dump.items():
            f.write('{} {}\n'.format(key.encode('utf-8'),val))



if __name__ == "__main__":
    # sorted_dscrp_key()
    make_label()




