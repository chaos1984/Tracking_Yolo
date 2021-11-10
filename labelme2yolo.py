import os
import numpy as np
import json
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from os import getcwd
 

 
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
 
 
def cropSquare(figdir):
    img = cv2.imread(figdir)
    w,h,c = img.shape
    l = min(w,h)
    dst = img[0:l, 0:l]
    dst = cv2.resize(dst,(640,640))
    cv2.imwrite(figdir,dst)
 


def ChangeToYolo5(files,labelme_path, txt_Name):
    if not os.path.exists(labelme_path+'tmp/'):
        os.makedirs(labelme_path+'tmp/')
    list_file = open(labelme_path+'tmp/%s.txt' % (txt_Name), 'w')
    for json_file_ in files:
        json_filename = labelme_path + json_file_ + ".json"
        imagePath = labelme_path + json_file_ + ".jpg"
        list_file.write('%s\n' % (imagePath))
        out_file = open('%s/%s.txt' % (labelme_path, json_file_), 'w')
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
            xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
            ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
            ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                cls_id = classes.index(label)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                print(json_filename, xmin, ymin, xmax, ymax, cls_id)


def ChangeToDeepsort(files,img_path = 'datasets/DAB/',anno_path = 'datasets/DAB/',cut_path = 'datasets/DAB_crop/'): 
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    # 获取文件夹中的文件
    # imagelist = os.listdir(img_path)
    # print(imagelist
    for image in files:
        # image_pre, ext = os.path.splitext(image)
        img_file = img_path + image+'.jpg'
        # image_pre, ext = os.path.splitext(img_file)
        img = cv2.imread(img_file)
        json_filename = img_file.replace(".jpg",".json")
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        # DOMTree = xml.dom.minidom.parse(xml_file)
        # collection = DOMTree.documentElement
        # objects = collection.getElementsByTagName("object")
 
       
        # root = tree.getroot()
        # if root.find('object') == None:
        #     return
        obj_i = 0
        for multi in json_file["shapes"]:
            obj_i += 1
            points = np.array(multi["points"])
            xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
            xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
            ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
            ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            cls = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                # cls_id = classes.index(label)
                b = [int(xmin), int(xmax), int(ymin), int(ymax)]
            img_cut = img[b[2]:b[3], b[0]:b[1], :]
            path = os.path.join(cut_path, cls)
            # 目录是否存在,不存在则创建
            mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
            mkdirlambda(path)
            cv2.imwrite(os.path.join(cut_path, cls, '{}_{:0>2d}.jpg'.format(image, obj_i)), img_cut)

        # for obj in root.iter('object'):
            
        #     cls = obj.find('name').text
        #     xmlbox = obj.find('bndbox')
        #     b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)),
        #          int(float(xmlbox.find('ymax').text))]
        #     img_cut = img[b[1]:b[3], b[0]:b[2], :]
        #     path = os.path.join(cut_path, cls)
        #     # 目录是否存在,不存在则创建
        #     mkdirlambda = lambda x: os.makedirs(x) if not os.path.exists(x) else True
        #     mkdirlambda(path)
        #     cv2.imwrite(os.path.join(cut_path, cls, '{}_{:0>2d}.jpg'.format(image_pre, obj_i)), img_cut)
        #     print("&&&&")
    return               
 
if __name__ =="__main__":
    filedir = ""

    classes = ["cushion","hub","seam","Front_cushion"]
    # 1.标签路径
    labelme_path = r"C:/Yoking/01_Study/Yolo/datasets/DAB/"
    isUseTest = True  # 是否创建test集
    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    # print(files)
    # if isUseTest:
    #     trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    # else:
    #     trainval_files = files
    # # split
    # train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)

    # wd = getcwd()
    # print(wd)

    # ChangeToYolo5(train_files,labelme_path, "train")
    # ChangeToYolo5(val_files, labelme_path,"val")
    # ChangeToYolo5(test_files, labelme_path,"test")

    ChangeToDeepsort(files,img_path = 'datasets/DAB/',anno_path = 'datasets/DAB/',cut_path = 'datasets/DAB_crop/')