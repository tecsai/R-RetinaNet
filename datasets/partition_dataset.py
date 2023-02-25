import os
import sys
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import random

DATASETS = ['IC15', 'IC13',
            'HRSC2016', 'DOTA', 'UCAS_AOD', 'NWPU_VHR' ,
            'GaoFenShip', 'GaoFenAirplane', 
            'VOC']


def bmpToJpg(file_path):
   for fileName in tqdm(os.listdir(file_path)):
       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
       im = Image.open(os.path.join(file_path,fileName))
       rgb = im.convert('RGB')      
       rgb.save(os.path.join(file_path,newFileName))

def del_bmp(root_dir=None):
    file_list = os.listdir(root_dir)
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            if f.endswith(".BMP") or f.endswith(".bmp"):
                os.remove(file_path)
                print( "File removed! " + file_path)
        elif os.path.isdir(file_path):
            del_bmp(file_path)




def partition_dataset(dataset_dir, dataset_group, train_prop=0.7, val_prop=0.2, test=0.1):
    basepath = dataset_dir
    train_dir = os.path.join(basepath, "train")
    val_dir = os.path.join(basepath, "val")
    test_dir = os.path.join(basepath, "test")
    
    for dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    train_image_dir = os.path.join(train_dir, "images")
    train_label_dir = os.path.join(train_dir, "labelTxt")
    val_image_dir = os.path.join(val_dir, "images")
    val_label_dir = os.path.join(val_dir, "labelTxt")
    test_image_dir = os.path.join(test_dir, "images")
    test_label_dir = os.path.join(test_dir, "labelTxt")
    for subdir in [train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    ''' 列出所有图像文件路径 '''
    imageslist = []
    for grp in dataset_group:
        image_path = os.path.join(basepath, grp, "images")
        images = os.listdir(image_path)
        for img in images:
            imageslist.append(os.path.join(image_path, img))
    random.shuffle(imageslist)

    ''' 校验图片对应的标签文件 '''
    for img in imageslist:
        lbl = img.replace("images", "labelTxt").replace("jpg", "txt")
        if not os.path.exists(lbl):
            print("No such file: {}".format(lbl))

    ''' 计算train, val, test的数量 '''
    total_imgs = len(imageslist)
    train_img_count = int(total_imgs * train_prop)
    val_img_count = int(total_imgs * val_prop)
    test_img_count = total_imgs - train_img_count - val_img_count
    print("total imagse: {}, train: {}, val: {}, test: {}".format(total_imgs, train_img_count, val_img_count, test_img_count))
    
    ''' 切分图像列表 '''
    train_imgs = []
    val_imgs = []
    test_imgs = []
    for i in range(total_imgs):
        if i < train_img_count:
            train_imgs.append(imageslist[i])
        elif i >= train_img_count and i < (train_img_count+val_img_count):
            val_imgs.append(imageslist[i])
        else:
            test_imgs.append(imageslist[i])
    print("total imagse: {}, train: {}, val: {}, test: {}".format(total_imgs, len(train_imgs), len(val_imgs), len(test_imgs)))

    for image in train_imgs:
        _, imgname = os.path.split(image)
        shutil.move(image, os.path.join(train_image_dir, imgname))
        shutil.move(image.replace("images", "labelTxt").replace("jpg", "txt"), os.path.join(train_label_dir, imgname.replace("jpg", "txt")))

    for image in val_imgs:
        _, imgname = os.path.split(image)
        shutil.move(image, os.path.join(val_image_dir, imgname))
        shutil.move(image.replace("images", "labelTxt").replace("jpg", "txt"), os.path.join(val_label_dir, imgname.replace("jpg", "txt")))

    for image in test_imgs:
        _, imgname = os.path.split(image)
        shutil.move(image, os.path.join(test_image_dir, imgname))
        shutil.move(image.replace("images", "labelTxt").replace("jpg", "txt"), os.path.join(test_label_dir, imgname.replace("jpg", "txt")))

    for grp in dataset_group:
        rmpath = os.path.join(basepath, grp)
        shutil.rmtree(rmpath)


if __name__ == '__main__':
    dataset_dir = "/2T/001_AI/1005_RotateRetinaNet/002_Datasets/IRInsu_20230222"
    dataset_group = ["group0", "group1"]
    partition_dataset(dataset_dir, dataset_group, )



