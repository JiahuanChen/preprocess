# encoding=utf-8
import sys
sys.path.insert(0,'/core1/data/home/xuqiang/mysql/')
# sys.path.insert(0,'/core1/data/home/shizhan/ai/utils/')
from mysql import MysqlOperator
import json
import jieba
import re
import collections

def get_data(select_sql):
    mysql = MysqlOperator()
    datas = list(mysql.select(select_sql))
    return datas

def parse_dscrp(dscrp):
    try: # json format
        dscrp = json.loads(dscrp)
    except: # save derectly
        new_dscrp = dscrp.strip('<ul><li>').strip('</li></ul>').split('</li><li>')
        if len(new_dscrp) == 1:
            return {}
        dscrp = {}
        for i in new_dscrp:
            try:
                key,val = re.split(':|：',i)
                dscrp[key.decode('utf-8')] = val.decode('utf-8')
            except:
                pass
    return dscrp

def analyse_dscrp(word_dict, datas):
    for idx,data in enumerate(datas):
            dscrp = data['dscrp']
            dscrp = parse_dscrp(dscrp)
            if len(dscrp) == 0:
                continue

            for key,val in dscrp.items():
                word_dict[key][val] += 1
                # if len(val) == 0:
                #     print key, val

def analyse_title(word_dict,datas):
    for idx,data in enumerate(datas):
        title = data['title']
        if title == '':
            continue
        seg_list = jieba.cut(title.decode('utf-8'), cut_all=True) # 精准分词
        for word in seg_list:
            word = word.strip()
            if len(word) == 0:
                continue
            word_dict[word] += 1

def analyse_color(word_dict,datas):
    for idx,data in enumerate(datas):
        color = data['color']
        if color == '':
            continue
        word_dict[color] += 1


def count_desc_and_title_and_color(batch_size, threshold):

    # read data from mysql
    # select_sql = '''
    #     select img_src.id src_id, dscrp, title
    #     from internal_website.image_source img_src 
    #     inner join internal_website.image img on img.src_id = img_src.id 
    #     inner join internal_website.object obj on obj.img_id = img.id
    #     where obj.src_src = 'jd_task_v2' and img.is_comment=0
    #     and obj.category_id > 0 and img_src.id >= {}
    #     limit {}
    # '''
    select_sql = '''
        select img_src.id src_id, dscrp, title, color
        from home.image_source img_src 
        where img_src.id >= {} and img_src.id < {}
        and src = 'TB_HOME'
    '''
    datas = get_data(select_sql.format(0, batch_size))
    assert len(datas) > 0
    word_dict_dscrp = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    word_dict_title = collections.defaultdict(lambda: 0)
    word_dict_color = collections.defaultdict(lambda: 0)
    done = 0
    # process the batch
    while len(datas) != 0 :
        # analyse_dscrp(word_dict_dscrp, datas)
        # analyse_title(word_dict_title, datas)
        analyse_color(word_dict_color, datas)
        datas = get_data(select_sql.format(start_id, start_id+batch_size))
        start_id += batch_size
        done += 1
        print done*batch_size , 'done'
        # if done >= 1:
        #     break

    # # save the title frequency
    # titleDictList = sorted(word_dict_title.items(), key=lambda x: x[1], reverse=True)
    # with open('./frequency/title_furniture.txt','w') as f:
    #     for (word, num) in titleDictList:
    #         if num <= threshold:
    #             break
    #         line = '%s : %s\n' % (word, str(num))
    #         f.write(line.encode('utf-8'))

    colorDictList = sorted(word_dict_color.items(), key=lambda x: x[1], reverse=True)
    with open('./frequency/color_furniture.txt','w') as f:
        for (word, num) in colorDictList:
            if num <= threshold:
                break
            line = '%s : %s\n' % (word, str(num))
            f.write(line.encode('utf-8'))

    # # save the dscrp frequency
    # with open('./frequency/dscrp_furniture.txt','w') as f:
    #     for key, val in word_dict_dscrp.items():
    #         dictList = sorted(val.items(), key=lambda x: x[1], reverse=True)
    #         line = key.encode('utf-8') + ':\n'
    #         for (word, num) in dictList:
    #             if num <= threshold:
    #                 break
    #             line += '  {} : {} \n'.format(word.encode('utf-8'), str(num))
    #         if line[-2] == ':':
    #             continue
    #         f.write(line)
    #         f.write('\n')

if __name__ == "__main__":
    # select_sql = '''
    #     select distinct(cat2) from home.image_source
    # '''
    # mysql = MysqlOperator()
    # datas = list(mysql.select(select_sql))
    # for idx, cat in enumerate(datas):
    #     print cat['cat2']
    #     count_desc_and_title_furniture(batch_size = 500000, threshold=10, cat2=cat['cat2'], cnt=idx)

    # count_desc_and_title_and_color(2000000, 5000)


    ##########
    # 每个cat统计
    with open('/data/data/jiahuan/label/furniture/furinture_0202/furniture_shop.txt','r') as f:
        lines = f.readlines()

    img_ids = [[] for i in range(45)]
    for line in lines:
        line = line.strip().split( '    ' )
        cat = map(int,line[3].split(','))
        try:
            cat_idx = cat.index(1)
        except:
            continue
        img_ids[cat_idx].append(line[0])

    for i in range(45):
        select_sql = '''
            select s.id src_id, dscrp, title, color
            from home.image_source s 
            inner join home.image i on i.src_id = s.id
            where i.id in ({})
            and s.src = 'TB_HOME'
        '''.format(','.join(img_ids[i]))
        datas = get_data(select_sql)
        assert len(datas) > 0
        word_dict_dscrp = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        word_dict_title = collections.defaultdict(lambda: 0)
        word_dict_color = collections.defaultdict(lambda: 0)
        analyse_dscrp(word_dict_dscrp, datas)
        analyse_title(word_dict_title, datas)

        # # save the title frequency
        titleDictList = sorted(word_dict_title.items(), key=lambda x: x[1], reverse=True)
        with open('./frequency/title_all_cat.txt','a') as f:
            f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            f.write('category {}: \n'.format(i+1))
            for (word, num) in titleDictList:
                if num <= 1000:
                    break
                line = '%s : %s\n' % (word, str(num))
                f.write(line.encode('utf-8'))

        # save the dscrp frequency
        new_dscrp = collections.defaultdict(lambda: 0)
        for key,val in word_dict_dscrp.items():

        with open('./frequency/dscrp_furniture.txt','w') as f:
            for key, val in word_dict_dscrp.items():
                dictList = sorted(val.items(), key=lambda x: x[1], reverse=True)
                line = key.encode('utf-8') + ':\n'
                for (word, num) in dictList:
                    if num <= threshold:
                        break
                    line += '  {} : {} \n'.format(word.encode('utf-8'), str(num))
                if line[-2] == ':':
                    continue
                f.write(line)
                f.write('\n')

            


