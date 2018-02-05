import sys
sys.path.insert(0,'/core1/data/home/xuqiang/mysql/')
from mysql import MysqlOperator
import json

def insert_data(sql,mysql):
    mysql.insert(sql,[])

def insert_into_sql():
    mysql = MysqlOperator()
    label_path = './furniture_labels_v1_all.txt'
    with open(label_path) as f:
        lines = f.readlines()
    categories = [200 for i in range(50)]
    imgid2cat = {}
    for line in lines:
        line = line.strip().split('    ')
        img_id = line[0]
        cat = int(line[2])
        if cat == 0:
            continue
        if categories[cat] >= 0:
            categories[cat] -= 1
        else:
            continue
        imgid2cat[line[0]] = line[2]
        sql = '''insert into home.image_to_mark_01_22   
        (select * from home.image where id = {})'''.format(img_id)
        insert_data(sql,mysql)
    json.dump(imgid2cat,open('imgid2cat_to_mark.json','w'))


if __name__ == "__main__":
    insert_into_sql()
    