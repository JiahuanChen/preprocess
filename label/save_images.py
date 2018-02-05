import cv2
import requests
cv_session = requests.Session()
cv_session.trust_env = False
import numpy as np

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

with open('./furniture_labels_v1_to_mark.txt') as f:
    lines = f.readlines()
cats = [5 for i in range(50)]
for line in lines:
    line = line.strip().split('    ')
    cat = int(line[2])
    if cats[cat] <= 0:
        continue
    path = line[1]
    img = cv_load_image(path, 'url')
    cv2.imwrite('/data/data/jiahuan/images/to_mark_example/{}_{}.jpg'.format(cat,cats[cat]), img)
    cats[cat] -= 1

