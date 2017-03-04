#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy
import sys
import os.path

def gray_face(filepath):

    cascade_path = 'face.xml'

    # 画像の読み込みとグレースケール化
    #img = cv2.LoadImage(filepath)
    img = cv2.imread(filepath)
    img_gray = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)

    # カスケード分類器の特徴の取得
    cascade = cv2.CascadeClassifier(cascade_path)

    # 顔認識
    facerect = cascade.detectMultiScale(
            img_gray,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))

    print filepath,len(facerect)

    if len(facerect) <= 0:
        print 'Couldm\'t be detected'
        sys.exit()

    rect = facerect[0]
    for r in facerect:
        if rect[2] < r[2]:
            rect = r

    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    cv2.imwrite('gray_face_' + os.path.basename(filepath),img_gray[y:y+h,x:x+w])

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Image path is not exist'
        sys.exit()

    gray_face(sys.argv[1])
