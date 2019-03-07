#USAGE
Simply run
```
$ python preproc_data.py
```
This code will parse the "labels.csv" to VOC2007-like or csv format.

#NOTE
1. Change "DATA_DIR" to the directory where your "object_detection_v1" and "object_detection_v1_2" locate.
2. "devkit_path" is the target path to place VOC2007-like format data.
3. Please copy all the images in "object_detection_v1" and "object_detection_v1_2" to "path/to/VOCdevkit/VOC2019/JPEGImages/" yourself, it will save time when re-preprocessing everytime.
4. Before converting to VOC2007-like format, please download the VOC2007 dataset at http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ .
