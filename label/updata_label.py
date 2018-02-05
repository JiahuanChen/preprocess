f_name = 'label_v1/category.txt'

with open(f_name,'r') as f:
    lines = f.readlines()

new_lines = []
for idx,line in enumerate(lines):
    cat = line.strip().split(' ')[0]
    new_lines.append('{} {}\n'.format(cat, idx+1))
with open(f_name,'w') as f:
    f.writelines(new_lines)