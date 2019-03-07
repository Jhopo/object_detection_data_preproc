import os
import sys
from os.path import join

import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
from xml.dom import minidom

DATA_DIR = '../'

devkit_path = '/media/jhopo/DATA/NTU/MSLAB/omnieyes/VOCdevkit2007/VOCdevkit/'
folder_name = 'VOC2019'
target_dir = join(devkit_path, folder_name)

image_sets = ['object_detection_v1', 'object_detection_v1_2']
label_sets = ['bus_grid', 'traffic_sign', 'logo', 'bus_stop', 'bus_station']


def try_mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

all_width, all_height = [], []
def check_annotation(id, xmin, ymin, xmax, ymax, width, height):
    all_width.append(xmax-xmin)
    all_height.append(ymax-ymin)

    return True

    if xmin < 0 or xmin > width or xmax < 0 or xmax > width or ymin < 0 or ymin > height or ymax < 0 or ymax > height:
        #print ('bounding error\n', id, xmin, ymin, xmax, ymax, width, height)
        return False

    if xmin >= xmax or ymin >= ymax:
        #print ('value error\n', id, xmin, ymin, xmax, ymax, width, height)
        return False

    if xmax - xmin <= 20 or ymax - ymin <= 20:
        #print ('small value error\n', id, xmin, ymin, xmax, ymax, width, height)
        return False

    if xmin == 0 or ymin == 0:
        #print ('small value error\n', id, xmin, ymin, xmax, ymax, width, height)
        return False

    return True


def parse_csv(data_dir):
    # "info_dict" is in the format of {id : {'path': path, 'size': (width, height), 'objects': [{'label': label, 'bbox': (xmin, ymin, xmax, ymax)}] }}
    info_dict = {}
    id_list = []
    label_set = set()

    for image_set in image_sets:
        info = pd.read_csv(join(data_dir, image_set, 'labels.csv'))

        paths = info["filename"].values
        widths = info["width"].values
        heights = info["height"].values
        labels = info["class"].values
        xmins = info["xmin"].values
        ymins = info["ymin"].values
        xmaxs = info["xmax"].values
        ymaxs = info["ymax"].values

        ids = [p.split('/')[-1].split('.jpg')[0] for p in paths]

        for idx, id in enumerate(ids):
            ret = check_annotation(id, xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx], widths[idx], heights[idx])
            if ret == True:
                if id not in info_dict:
                    info_dict[id] = {'path': None, 'size': None, 'objects':[]}
                    id_list.append(id)

        for idx, id in enumerate(ids):
            ret = check_annotation(id, xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx], widths[idx], heights[idx])
            if ret == True:
                info_dict[id]['path'] = join(target_dir, 'JPEGImages', '{}.jpg'.format(id))
                info_dict[id]['size'] = (widths[idx], heights[idx])
                info_dict[id]['objects'].append( {'label': labels[idx], 'bbox': (xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx])} )
                label_set.add(labels[idx])

    return info_dict, id_list, label_set


def split_train_val(info_dict, id_list):
    shot_dict = {} #  format -> {shot: [id]}
    for id in id_list:
        shot, frame = id.split('_frame')[0], id.split('_frame')[1]

        if shot not in shot_dict:
            shot_dict[shot] = [id]
        else:
            shot_dict[shot].append(id)

    trainval_id, test_id = [], []
    for shot in shot_dict:
        num_frame = len(shot_dict[shot])
        num_trainval = int(num_frame*3/4)
        trainval_id += shot_dict[shot][:num_trainval]
        test_id += shot_dict[shot][num_trainval:]

    return trainval_id, test_id


def convert_to_voc_format(info_dict, id_list, trainval_id, test_id):
    # create directories
    try_mkdir(join(target_dir))
    try_mkdir(join(target_dir, 'Annotations'))
    try_mkdir(join(target_dir, 'ImageSets', 'Main'))
    try_mkdir(join(target_dir, 'JPEGImages'))

    # copy image files, please do it yourself.
    pass

    # generate image txt
    trainval_test_id = [trainval_id, test_id]
    for idx, image_set in enumerate(['trainval', 'test']):
        id_list = trainval_test_id[idx]

        with open(join(target_dir, 'ImageSets', 'Main', '{}.txt'.format(image_set)), 'w') as f:
            for id in id_list:
                f.write('{}\n'.format(id))

    # generate annotation xml
    for id in info_dict:
        info = info_dict[id]
        width, height = info['size']

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = folder_name
        ET.SubElement(annotation, "filename").text = '{}.jpg'.format(id)

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = '3'

        for object in info['objects']:
            label = object['label']
            xmin, ymin, xmax, ymax = object['bbox']

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = label
            ET.SubElement(obj, "truncated").text = '0'
            ET.SubElement(obj, "difficult").text = '0'
            ET.SubElement(obj, "pose").text = 'Unspecified'

            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(xmin)
            ET.SubElement(bbox, "ymin").text = str(ymin)
            ET.SubElement(bbox, "xmax").text = str(xmax)
            ET.SubElement(bbox, "ymax").text = str(ymax)

        xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
        with open(join(target_dir, 'Annotations', '{}.xml'.format(id)), "w") as f:
            f.write(xmlstr)


def convert_to_csv_format(info_dict, id_list, trainval_id, test_id):
    # generate class mapping csv
    with open('./class.csv', 'w') as f:
        for idx, label in enumerate(label_sets):
            f.write('{},{}\n'.format(label, idx))

    def generate_csv(trainval_id, filename):
        with open('./{}.csv'.format(filename), 'w') as f:
            for id in trainval_id:
                info = info_dict[id]
                path = info['path']
                width, height = info['size']

                for object in info['objects']:
                    label = object['label']
                    xmin, ymin, xmax, ymax = object['bbox']

                    f.write('{},{},{},{},{},{}\n'.format(path, xmin, ymin, xmax, ymax, label))

    generate_csv(trainval_id, 'train')
    generate_csv(test_id, 'test')

if __name__ == '__main__':

    # Parse labels.csv under directories "object_detection_v1" and "object_detection_v1_2".
    info_dict, id_list, label_set = parse_csv(DATA_DIR)

    # Split data into training (0.75) and testing (0.25) set.
    trainval_id, test_id = split_train_val(info_dict, id_list)

    # Convert to VOC2007 data format.
    convert_to_voc_format(info_dict, id_list, trainval_id, test_id)

    # Convert to CSV data format.
    convert_to_csv_format(info_dict, id_list, trainval_id, test_id)
