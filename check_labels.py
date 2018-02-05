import collections

with open('taobao_home_dict.txt','r') as f:
    lines = f.readlines()

labels = collections.defaultdict(lambda: 0)
for line in lines:
    l = line.strip().split(':')[1][:-1].strip('"')
    labels[l] += 1

print len(labels)
for key in labels.keys():
    print key,

with open('/core1/data/home/shizhan/jiahuan/furniture/label/label/label_v1/category.txt','w') as f:
    to_write = "{} {}\n"
    for idx,key in enumerate(labels.keys()):
        f.write(to_write.format(key, idx+1))
