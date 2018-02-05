#coding=utf-8
import sys
sys.path.insert(0,'/core1/data/home/xuqiang/mysql/')
from mysql import MysqlOperator
import re
import collections
import json
import cv2
import requests
import time
cv_session = requests.Session()
cv_session.trust_env = False
import numpy as np
import random
from random import shuffle
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
font_path = '/core1/data/home/shizhan/jiahuan/color/code/color_predict/SimHei.ttf'
prop = mfm.FontProperties(fname=font_path)

def cv_load_image(in_, type_='path'):
    '''
    Return
        image: opencv format np.array. (C x H x W) in BGR np.uint8
    '''
    if type_ == 'url':
        img_nparr = np.fromstring(cv_session.get(in_).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(in_, cv2.IMREAD_COLOR )
    return img 

def get_data(select_sql):
    mysql = MysqlOperator()
    datas = list(mysql.select(select_sql))
    return datas

def load_cat(cat_path):
    with open(cat_path,'r') as f:
        lines = f.readlines()
    cat_dict = {}
    for line in lines:
        line = line.strip().split(' ')
        cats = line[0].split('/')
        idx = int(line[1])
        for cat in cats:
            cat_dict[cat.decode('utf-8')] = idx

    val2cat = collections.defaultdict(lambda: '')
    for key,val in cat_dict.items():
        val2cat[val] += key
    return cat_dict,val2cat

def load_attr(cat_path):
    data = json.load(open(cat_path,'r'))
    attr_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for key,val in data.items():
        key_out,key_in = key.split('_')
        attr_dict[key_out][key_in] = val
    return attr_dict,data

def visualize(data,multilabel,single_label_word):
    print 'information:'
    print 'pid:',data['pid'],'cat:',data['cat2']
    print 'slink:',data['slink']
    print 'path:',data['path']
    print '-------------------'

    img = cv_load_image(data['path'], 'url')
    plt.imshow(img[...,(2,1,0)])
    plt.title(data['title'].decode('utf-8'), fontproperties=prop, color = 'r', fontsize = 12)
    plt.text(0,30+img.shape[0],u'multilabel:'+multilabel[:-1], fontproperties=prop, fontsize = 10)
    plt.text(0,60+img.shape[0],u'singlelabel:'+single_label_word, fontproperties=prop, fontsize = 10)
    plt.axis('off')
    plt.show()

def validate_title(attr_validation,dscrp,single_label):
    if single_label == 44 :
        if '包含组件'.decode('utf-8') not in dscrp:
            return True
        else :
            return False
    if single_label not in attr_validation:
        return True
    for key in attr_validation[single_label]:
        key = key.decode('utf-8')
        if key in dscrp.keys():
            return True
    return False

def make_txt(cat_num,attr_num,target):

    # load category
    cat_path = './label_v2/category_v2.txt'
    attr_path = './label_v1/attribute.json'
    cat_dict,val2cat = load_cat(cat_path)
    attr_dict, _ = load_attr(attr_path)

    attr_validation ={
                1:{'门数量':1,'抽屉导轨节数':1},
                2:{'附加功能'},
                4:{'沙发组合形式':1, '适用人数':1, '几人座':1},
                5:{'是否可旋转':1, '是否可升降':1},
                6:{'是否可折叠':1},
                7:{'靠背高度':1, '是否有扶手':1},
                10:{'门数量':1},
                11:{'包含组件':1},
                12:{'斗数':1, '层数':1},
                13:{'门数量':1, '开合方式':1},
                14:{'是否含镜灯':1, '台面类型':1},
                15:{'衣柜类型':1},
                16:{'柜子类型':1},
                23:{'是否组装':1, '安装说明详情':1},
                29:{'开合方式':1, '门板材质':1, '边框材质':1},
                30:{'窗户打开方式':1, '玻璃类型':1},
                31:{'槽数':1},
                34:{'扶手':1, '是否含龙头':1, '沐浴桶类型':1},
                35:{'安装方式':1},
                36:{'是否内设蒸汽':1, '淋浴房类型':1, '开启方式':1, '开合方式':1, '是否含底盆':1},
                37:{'坐便器类型':1, '坐便冲水量':1, '坐便器冲水方式':1, '最小坑距':1, '冲水按键类型':1},
                38:{'每片宽度(mm)':1, '每片长度(mm)':1},
                41:{'摆件类型':1},
                42:{'花器种类':1, '摆放空间':1, '仿真花类型':1},
                43:{'套餐类型':1, '照片框数':1},
                   # 屏风 - 要判断title中有没有’柜’，而且dscrp没有包含组件
                }

    # status = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    # multilabel_dict = collections.defaultdict(lambda: 0)
    # multilabel_single_count = np.zeros(cat_num+1, dtype='int')
    cat_count = [[0,0] for i in range(cat_num+1)]
    failed = 0
    attr_satus = collections.defaultdict(lambda: 0)
        
    sel_id, maxid = 0, 9404256
    offset = 5000000
    while sel_id < maxid:
        # load data from mysql
        ### 取训练数据
        # select_sql = '''
        # select s.pid, s.title, s.dscrp, s.color, s.cat2, concat('http://',i.path) path, s.slink,
        # i.id img_id, i.is_comment, i.width, i.height
        # from home.image_source s
        # inner join home.image i on i.src_id = s.id
        # where s.pid not in (select id from home.clean_data ) 
        # and i.id in (select id from home.image_val)
        # and s.src = 'TB_HOME' and i.id > {} and i.id < {} and i.is_comment = 0
        # '''.format(sel_id, sel_id + offset)
        # 取标注数据
        select_sql = '''
        select s.pid, s.title, s.dscrp, s.color, s.cat2, concat('http://',i.path) path, s.slink,
        i.id img_id, i.is_comment, i.width, i.height
        from home.image_source s
        inner join home.image i on i.src_id = s.id
        inner join home.object o on i.id = o.img_id 
        where i.status = 4 and o.box_status = 0 
        and i.id > {} and i.id < {}
        '''.format(sel_id, sel_id + offset)
        
        datas = get_data(select_sql)
        sel_id += offset
        print len(datas)
        # shuffle(datas)

        cat_time = 0
        attr_time = 0
        io_time = 0
        with open(target,'a') as f:
            for idx,data in enumerate(datas):
                if idx % 100000 == 0:
                    print idx, 'done'
                    print 'cat_time:',cat_time,'attr_time:',attr_time,'io_time:',io_time
                    cat_time = 0
                    attr_time = 0
                    io_time = 0
                stime = time.time()
                bbox = ','.join(map(str,[0,0,data['width']-1,data['height']-1]))
                title = data['title'].decode('utf-8')
                cat = data['cat2']
                cat_vector = [0 for i in range(cat_num+1)]

                ### category, title 匹配多个关键词
                # multilabel = ''
                # for key,val in cat_dict.items():
                #     if re.search(key,title):
                #         # print key
                #         cat_vector[val] = 1
                #         multilabel += key
                #         multilabel += '+' 
                # status[cat][sum(cat_vector)] += 1
                # if sum(cat_vector) >= 2:
                #     multilabel_dict[multilabel[:-1]] += 1
                # elif sum(cat_vector) == 1:
                #     multilabel_single_count += np.array(cat_vector,dtype='int')

                ### category, title 匹配频率最高的关键词
                cat_vector_freq = [0 for i in range(cat_num+1)]
                
                for key,val in cat_dict.items():
                    cat_vector_freq[val] += len( re.findall(key, title) )
                
                # deal with 床头柜
                if cat_vector_freq[cat_dict['床头柜'.decode('utf-8')]] != 0:
                    single_label = cat_dict['床头柜'.decode('utf-8')]
                # deal with 梳妆台
                elif cat_vector_freq[cat_dict['梳妆台'.decode('utf-8')]] != 0:
                    single_label = cat_dict['梳妆台'.decode('utf-8')]
                # deal with 折叠床
                elif cat_vector_freq[cat_dict['折叠床'.decode('utf-8')]] != 0:
                    single_label = cat_dict['折叠床'.decode('utf-8')]
                elif cat_vector_freq[cat_dict['更衣柜'.decode('utf-8')]] != 0:
                    single_label = cat_dict['更衣柜'.decode('utf-8')]
                # deal with 门
                elif cat_vector_freq[cat_dict['门'.decode('utf-8')]] != 0 and \
                (re.search('柜'.decode('utf-8'), title) != None or\
                 re.search('橱'.decode('utf-8'), title) != None ) :
                    cat_vector_freq[cat_dict['门'.decode('utf-8')]] = 0
                    single_label = cat_vector_freq.index(max(cat_vector_freq))
                # deal with 屏风
                elif cat_vector_freq[cat_dict['屏风'.decode('utf-8')]] != 0 and \
                re.search('柜'.decode('utf-8'), title) == None:
                    single_label = cat_dict['屏风'.decode('utf-8')]
                else:
                    single_label = cat_vector_freq.index(max(cat_vector_freq))

                #validate the title with dscrp
                dscrp = json.loads(data['dscrp'])
                if not validate_title(attr_validation,dscrp,single_label):
                    cat_count[ 0 ][ data['is_comment'] ] += 1
                    continue

                assert single_label <= cat_num
                cat_vector = ','.join(map(str,[1 if _ == single_label-1 else -1 for _ in range(cat_num)]))
                single_label_word = val2cat[single_label]
                cat_count[ single_label ][ data['is_comment'] ] += 1
                
                cat_time += (time.time() - stime)
                stime = time.time()
    
                # attribute, 用description匹配
                attr_vector = [-1 for _ in range(attr_num)]
                for key,val in dscrp.items():
                    if key in attr_dict:
                        for sub_key in attr_dict[key]:
                            if re.search(sub_key, val):
                                attr = attr_dict[key][sub_key]
                                assert attr <= attr_num and attr >= 0
                                # print sub_key , 'found in', val
                                attr_satus[attr] += 1
                                attr_vector[ attr ] = 1 
                attr_vector = ','.join(map(str,attr_vector))

                attr_time += (time.time() - stime)
                stime = time.time()
    
                # print single_label_word
                # visualize(data,multilabel,single_label_word)
                # if sum(cat_vector) == 1:
                #     print '!!!'
                #     visualize(data,multilabel,single_label_word)
                # visualize(data,multilabel,single_label_word)
                # to_write = '{}    {}    {}\n'.format(data['img_id'],data['path'],\
                #     ','.join(['1' if i == single_label else '0' for i in range(cat_num+1)]))

                # format:
                # imgid    path    bbox    cat_vector    attr_vector
                to_write = '{}    {}    {}    {}    {}\n'.\
                format(         data['img_id'], \
                                data['path'], \
                                bbox,\
                                cat_vector, \
                                attr_vector, )
                f.write(to_write)
                io_time += (time.time() - stime)
                stime = time.time()
    
    # all_status = collections.defaultdict(lambda: 0)
    # for key,val in status.items():
    #     print '{:16}:'.format(key),
    #     for sub_key,sub_val in val.items():
    #         print  '{}:{},'.format(sub_key,sub_val),
    #         all_status[sub_key] += sub_val
    #     print ''
    # print '------'
    # for key,val in all_status.items():
    #     print  '{}:{},\t'.format(key, val),

    # print ''
    # print '------'
    # dictList = sorted(multilabel_dict.items(), key=lambda x: x[1], reverse=True)
    # for key,val in dictList:
    #     print  '{}:{},\t'.format(key.encode('utf-8'), val)


    # print 'multilabel'
    # print status
    # for idx in range(1,cat_num+1):
    #     print idx, val2cat[idx].encode('utf-8'), multilabel_single_count[idx]
    # print 'single label'
    
    #### analyze singlelabel
    print 'failed',sum(cat_count[0])
    shop = 0
    comment = 0
    for idx in range(1,cat_num+1):
        print idx, val2cat[idx].encode('utf-8'), cat_count[idx][0]+cat_count[idx][1]
        shop += cat_count[idx][0]
        comment += cat_count[idx][1]
    print shop, comment

    # analyze attr
    for key_out in attr_dict.keys():
        print key_out.encode('utf-8'),':'
        for key_in in attr_dict[key_out].keys():
            print '    {}:'.format(key_in.encode('utf-8')), attr_satus[attr_dict[key_out][key_in]]


def visualize_txt(label_path):
    cat_path = './label_v2/category_v2.txt'
    cat_dict,val2cat = load_cat(cat_path)
    attr_path = './label_v1/attribute.json'
    _,data = load_attr(attr_path)
    idx2attr = {}
    for key,val in data.items():
        idx2attr[val] = key
    with open(label_path) as f:
        lines = f.readlines()

    # random.seed(1)
    shuffle(lines)
    for line in lines:
        line = line.strip().split('    ')
        bbox = line[2]
        bbox = map(float, bbox.split(','))
        bbox = map(int, bbox)
        cat_vector = map(int, line[3].split(','))
        try:
            cat = val2cat[cat_vector.index(1)+1]
            cat_idx = cat_vector.index(1)+1
        except:
            cat = '没有匹配'.decode('utf-8')
            cat_idx = -1
        # attr_vector = map(int, line[4].split(','))
        # attr = ""
        # for idx,val in enumerate(attr_vector):
        #     if val == 1:
        #         attr += idx2attr[idx]
        #         attr += '+'
        # attr = attr[:-1]
        # if attr == '':
        #     print '!'
        #     continue
        # if cat != '窗'.decode('utf-8'):
        #     continue
        if cat_idx != 38:
            continue
        img_id = line[0]
        select_sql = '''
        select * from home.image_source s 
        inner join home.image i on i.src_id = s.id
        where i.id = {}
        '''.format(img_id)
        data = get_data(select_sql)[0]
        path = line[1]
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print img_id,path,data['path']
        print data['slink']
        print data['title']
        dscrp = json.loads(data['dscrp'])
        for key,val in dscrp.items():
            print key,':',val
        # print bbox
        # print attr

        img = cv_load_image(path, 'url')
        print img.shape, bbox
        plt.imshow(img[...,(2,1,0)])
        rect = plt.Rectangle((  bbox[0], 
                                bbox[1] ),
                                bbox[2],
                                bbox[3], 

                                fill=False, linewidth=1.5)
        plt.gca().add_patch(rect) 
        plt.text(0,-30,cat, fontproperties=prop, fontsize = 10)
        # plt.text(0,30+img.shape[0],attr, fontproperties=prop, fontsize = 10)
        
        plt.axis('off')
        plt.show()

def fix_txt(old_txt,new_txt):
    with open(old_txt,'r') as f:
        lines = f.readlines()
    shuffle(lines)
    with open(new_txt,'w') as f:
        for line in lines:
            line = line.strip().split('    ')
            bbox = line[2].split(',')
            bbox[2],bbox[3] = bbox[3], bbox[2]
            line[2] = ','.join(bbox)
            line = '    '.join(line)
            line += '\n'
            f.write(line)

def balance_category(in_name, out_name, max_num):
    with open(in_name,'r') as f:
        lines = f.readlines()
    shuffle(lines)
    new_lines = []

    categories_count = [max_num for i in range(45)]
    ###### add special rules here

    for idx,line in enumerate(lines):
        if idx%100000 == 0:
            print idx
        flag = line.strip().split('    ')
        cat_vector = map(int, flag[3].split(','))
        try:
            cat = cat_vector.index(1)
        except:
            cat = -1
        if cat == -1:
            continue
        if categories_count[cat] > 0: 
            new_lines.append(line)
            categories_count[cat] -= 1
    shuffle(new_lines)
    with open(out_name,'w') as f:
        f.writelines(new_lines)
    
def make_train_val(f_name, train_num, val_num):
    with open(f_name,'r') as f:
        lines = f.readlines()
    shuffle(lines)
    with open(f_name[:-4]+'_train.txt','w') as f:
        f.writelines(lines[:train_num])
    with open(f_name[:-4]+'_val.txt','w') as f:
        f.writelines(lines[train_num:train_num+val_num])

def make_val_sql(val_f_name):
    with open(val_f_name,'r') as f:
        lines = f.readlines()
        to_insert = ''
        for idx,line in enumerate(lines):
            if idx % 5000 == 0:
                print idx
            flag = line.strip().split('    ')
            to_insert += '({}),'.format(flag[0])

        mysql = MysqlOperator()
        insert_sql = '''
        insert into home.image_val values {}
        '''
        print 
        mysql.insert(insert_sql.format( to_insert[:-1] ),[])

if __name__ == "__main__":
    make_txt(cat_num = 45, attr_num = 154,\
    target = '/data/data/jiahuan/label/furniture/furinture_0202/detection.txt')
    # visualize_txt('/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop_balance_val.txt')

    #########
    ### fix bbox
    # fix_txt('/data/data/jiahuan/label/furniture/furniture_all_single.txt',\
    #     '/data/data/jiahuan/label/furniture/furniture_all_single_fixed.txt')
    # balance_category('/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop.txt',
    #     '/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop_balance.txt',
    #     50000)
    # make_train_val('/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop_balance.txt',500000,50000)
    # make_val_sql('/data/data/jiahuan/label/furniture/furinture_0130/furniture_shop_balance_val.txt')
