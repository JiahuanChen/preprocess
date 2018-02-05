#! /usr/bin/env python
#coding=utf-8
from random import shuffle
import pprint
import json
import lmdb
import numpy as np
import cv2
import sys
import os
import random
import multiprocessing
import time
import requests
from collections import defaultdict
cv_session = requests.Session()
cv_session.trust_env = False

import matplotlib.pyplot as plt
## to support chinese characters
plt.rcParams['font.sans-serif']=['Songti SC']

caffe_path = '/core1/data/home/liuhuawei/detection/caffe/python/'
sys.path.insert(0, caffe_path)
from caffe.proto import caffe_pb2

# sys.path.insert(0, '../utils/')
sys.path.insert(0,'/core1/data/home/liuhuawei/evalution_ai/utils/')
from arg_parser import config, init_mysql
from data_augmentation import scale_bbox

class AnnoData(object):
    def __init__(self, **kwargs):
        self.mysql = init_mysql(section_name=kwargs.get('mysql_conf_sec'), conf_file=kwargs.get('conf_file', 'pretrain.cfg'))
 
    def add_jd_and_tb_image(self, dp_txt, num_attrs, num_cats, shuffle=True, \
            output_label_lmdb_path='',output_data_lmdb_path='', num_train=None, num_thread=1, begin_index=0):
        all_data = []
        with open(dp_txt, 'r') as f_dp:
            data = f_dp.readlines()
            if shuffle:
                random.shuffle(data)
                print 'shuffled'                
            if num_train is not None:
                data = data[:num_train]    
            print 'Num of examples: {}'.format(len(data))
            for item in data:
                flags = item.strip('\n').split('    ')
                # print flags
                assert len(flags)==5
                img_path = flags[begin_index]
                bbox = map(float, flags[begin_index+1].split(','))
                bbox = map(int, bbox)
                attrs = map(float, flags[begin_index+3].split(','))
                attrs = map(int, attrs)
                cats = map(float, flags[begin_index+2].split(','))
                cats = map(int, cats)
                assert len(bbox)==4 
                assert len(attrs)==num_attrs ,'{}!={}'.format(len(attrs),num_attrs)
                assert len(cats)==num_cats
                all_data.append((img_path, bbox, attrs, cats))          

        stats = defaultdict(int)  
        attr_stats = np.zeros(len(attrs),dtype='int')   
        all_img_path = []
        all_bboxes = [] 
        all_cats = []   
        all_attrs = []
        for idx, data in enumerate(all_data):
            if idx % 1000 == 0:
                print 'Processing %d/%d!!!!' % (idx, len(all_data))
            assert len(data)==4
            all_img_path.append(data[0])
            bbox = data[1]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            all_bboxes.append([xmin, ymin, xmax, ymax])
            # attrs = map(int, data['attributes_id'].strip().split(','))
            attrs = data[2]
            attrs = [0 if k==-1 else 1 for k in attrs]
            all_attrs.append(attrs)

            cats = np.array(data[3], dtype=int)
            cats_ind = np.where(cats==1)[0]
            if len(cats_ind) != 1:
                stats[-1] += 1
                all_cats.append(-1)
            else:
                stats[cats_ind[0]] += 1
                if  cats_ind[0] == 37:
                    print '~~~~~~~~~'
                attr_stats += np.array(attrs,dtype='int')
                all_cats.append(cats_ind[0])   


        print 'categories statisticals is: ', stats
        print attr_stats   
        with open('/core1/data/home/shizhan/jiahuan/furniture/label/label/label_v1/category.txt' ) as f:
            cat_list = f.readlines()
        with open('/core1/data/home/shizhan/jiahuan/furniture/label/label/label_v1/attribute.txt' ) as f:
            attr_list = f.readlines()
        attr_dict = {}
        for data in attr_list:
            data = data.strip().split(' ')
            attr_dict[data[1]] = data[0]

        for key,val in stats.items():
            print cat_list[key].split(' ')[0],val

        print 'attr'
        for idx,val in enumerate(attr_stats):
            print attr_dict[str(idx)], val
     
        # create_label_lmdb(np.array(all_attrs, dtype=np.uint8), output_label_lmdb_path)   
        # create_data_lmdb(all_img_path, all_bboxes, all_cats, output_data_lmdb_path, num_thread=num_thread)   
        
def create_label_lmdb(label_contents, output_lmdb_path):
    # print label_contents
    print output_lmdb_path
    assert not os.path.exists(output_lmdb_path)
    print label_contents.shape
    num_instance, num_labels = label_contents.shape
    X = np.zeros((num_instance, num_labels, 1, 1), dtype=np.uint8)
    y = np.zeros(num_instance, dtype=np.int64)
    map_size = X.nbytes * 1000
    env = lmdb.open(output_lmdb_path, map_size=map_size)

    X[:, :, 0, 0] = label_contents
    # Check to see if the contents are well populated within the expected range
    print X.shape   # Check to see if X is of shape N x M x 1 x 1     

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        cnt = 0
        for i in range(num_instance):
            datum = caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tostring()    # or .tobytes() if numpy < 1.9 
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)
            txn.put(str_id, datum.SerializeToString())
            if cnt % 1000 == 0:
                print 'Done ', cnt, 'images'
            cnt += 1
        print 'Totally done ', cnt, 'images'  

def create_data_lmdb(all_img_path, all_bboxes, all_cats, output_data_lmdb_path, num_thread=8) :
    '''
    Read file list for training and validation, write instance into lmdb.
    Input:
        fl_path(str): /path/to/file/list
        lmdb_path(str): /path/to/lmdb
        prefix(str): path prefix for images in fl_path
    '''
    assert len(all_img_path)==len(all_bboxes)==len(all_cats)
    q_in = [multiprocessing.Queue(1024) for i in range(num_thread)]
    q_out = multiprocessing.Queue(1024)
    read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) \
                    for i in range(num_thread)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, output_data_lmdb_path))
    write_process.start()

    for i, item in enumerate(zip(all_img_path, all_bboxes, all_cats)):
        q_in[i % len(q_in)].put((i, item))
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()

    q_out.put(None)
    write_process.join()

def write_worker(q_out, output_data_lmdb_path):
    pre_time = time.time()
    count = 0
    assert not os.path.exists(output_data_lmdb_path)
    env = lmdb.open(output_data_lmdb_path, map_size=1e12)
    txn = env.begin(write=True)
    
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, str_id, item = deq
            buf[i] = (str_id, item)
        else:
            more = False
        while count in buf:
            str_id, datum = buf[count]
            del buf[count]
            if str_id is not None:
                txn.put(str_id, datum)
            if count % 1000 == 0:
                txn.commit()
                env.sync()
                txn = env.begin(write=True)                    
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1  
    txn.commit()
    env.sync()                              

def read_worker(q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        img_path = item[0]
        bbox = item[1]
        cat_id = item[2]
        img = transform_img(img_path, bbox, resize_h=256, resize_w=256)
        if img is None:
            print img_path, 'is None'
            img = np.zeros((3,256,256))
            datum = get_datum(img,-1)
        else:
            datum = get_datum(img, int(cat_id))
        str_id = '{:08}'.format(i)
        q_out.put((i,str_id,datum.SerializeToString()))

def get_datum(img, label):
    '''
    Store image and label in caffe Datum.
    Input:
        img(numpy.ndarray()): C x H x W, BGR format
        label(int): label
    Return:
        datum(caffe_pb2.Datum): caffe datum instance 
    '''
    datum = caffe_pb2.Datum()
    datum.channels = img.shape[0]
    datum.height = img.shape[1]
    datum.width = img.shape[2]
    datum.label = label
    datum.data = img.tostring()
    return datum 

def transform_img(img_name, bbox, resize_h, resize_w):
    if img_name[:4] == 'http':
        type_ = 'url'
    else:
        type_ = 'path'
    img = cv_load_image(img_name, type_)
    img_o = img
    cnt = 0
    if img is None:
        print img_name
        return img
    # print bbox
    # print len(bbox)
    # raise
    bbox = np.array(bbox).reshape((1, -1))
    bbox = scale_bbox(bbox, 1.1).astype(int)[0]
    H, W, _ = img.shape
    bbox[2] = min(bbox[2], W)
    bbox[3] = min(bbox[3], H)
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    img = cv2.resize(img, (resize_w, resize_h))
    img = img.transpose((2, 0, 1))

    # if img_o.shape[0] != img_o.shape[1]:
    #     plt.subplot(1,2,1)
    #     plt.imshow(img_o[...,(2,1,0)])
    #     plt.subplot(1,2,2)
    #     plt.imshow(img.transpose(1,2,0)[...,(2,1,0)])
    #     plt.show()
    return img

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]        

def cv_load_image(in_, type_='path'):
    '''
    Return
        image: opencv format np.array. (C x H x W) in BGR np.uint8
    '''
    if type_ == 'url':
        img_nparr = np.fromstring(cv_session.get(in_).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(in_, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img 

if __name__ == '__main__':
    ad = AnnoData(mysql_conf_sec='mysql_internal')
    # task_type = 'test'
    task_type = 'val'
    # task_type = 'train'
    num_attrs = 154
    num_cats = 45
    dp_txt = '/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop_balance_{}.txt'.format(task_type)
    output_label_lmdb_path = \
    '/data/data/jiahuan/lmdb/furniture/balance_2/{}/{}_label_lmdb'.\
    format(task_type, task_type)
    output_data_lmdb_path = \
    '/data/data/jiahuan/lmdb/furniture/balance_2/{}/{}_data_lmdb'.\
    format(task_type, task_type)
    
    # dp_txt = '/data/data/jiahuan/label/furniture/fake.txt'
    # output_label_lmdb_path = \
    # '/data/data/jiahuan/lmdb/furniture/test/{}/{}_label_lmdb'.\
    # format(task_type, task_type)
    # output_data_lmdb_path = \
    # '/data/data/jiahuan/lmdb/furniture/test/{}/{}_data_lmdb'.\
    # format(task_type, task_type)
    print dp_txt
    ad.add_jd_and_tb_image(dp_txt, num_attrs, num_cats, shuffle=True, \
            output_label_lmdb_path=output_label_lmdb_path,
            output_data_lmdb_path=output_data_lmdb_path, 
            num_train=None, num_thread=4, begin_index=1)    
    