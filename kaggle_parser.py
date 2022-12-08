from xml.dom import minidom
import cv2
import numpy as np
import os


def parseXML(filename):
    num = 0
    kaggle_path = "kaggle\\annotations\\"
    file = minidom.parse(kaggle_path + filename)

    imgPath = file.getElementsByTagName('filename')[0].childNodes[0].data
    imgPath = "kaggle\\images\\" + imgPath
    img = cv2.imread(imgPath)

    objects = file.getElementsByTagName('object')
    for elem in objects:
        maskType = str(elem.getElementsByTagName('name')[0].childNodes[0].data)
        bndbox = elem.getElementsByTagName('bndbox')[0]
        xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
        if not (maskType == "with_mask" or maskType == "without_mask"):
            continue
        cropped = img[ymin:ymax, xmin:xmax]
        saveName = "dataset\\" + maskType + "\\" + filename + str(num) + ".png"
        print(saveName)
        cv2.imwrite(saveName, cropped)
        num += 1


kaggle_path = "kaggle\\annotations\\"
for filename in os.listdir(kaggle_path):
    if not filename.endswith('.xml'):
        continue
    parseXML(filename)
