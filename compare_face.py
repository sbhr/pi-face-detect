#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import cv2
import os
import sys
import time

# 比較方法と比較画像を受け取る
def compare(detector_name,extractor_name,matcher_name,face_image):

    # キーポイントの検出
    detector = cv2.FeatureDetector_create(detector_name)
    keypoints1 = detector.detect(face_image)

    # 画像データの特徴量
    descripter = cv2.DescriptorExtractor_create(extractor_name)
    k1,d1 = descripter.compute(face_image,keypoints1)

    # matcher準備
    matcher = cv2.DescriptorMatcher_create(matcher_name)

    min_dist = 100000
    users = {}

    # 比較元画像読み込み
    imgdir = '/home/pi/fd/user'
    files = os.listdir(imgdir)
    for file in files:
        print 'Target:',file

        #画像はグレースケール変換済み
        test_img = cv2.imread(os.path.join(imgdir,file))

        # キーポイントの検出
        keypoints2 = detector.detect(test_img)
        k2,d2, = descripter.compute(test_img,keypoints2)

        # キーの一致度合いを調べる
        try:
            matches = matcher.match(d1,d2)
        except:
            continue

        dist = [m.distance for m in matches]

        if len(dist) == 0:
            continue

        min_dist = min(min(dist),min_dist)
        print 'dist:',min_dist
        users[min_dist] = file[:len(file)-4]

    print 'Detect:',users[min(users)]
    return min_dist

def print_time(sec):

    hour = int(sec / 3600)
    minutes = int((sec % 3600)/60)
    second = int(((sec % 3600) % 60))
    if hour != 0:
        txt = '%d時間%d分%d秒かかりました' % (hour,minutes,second)
    elif minutes != 0:
        txt = '%d分%d秒かかりました' % (minutes,second)
    else:
        txt = '%d秒かかりました' % (second)
    print txt
    return txt

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Couldn\'t exist image file'
        sys.exit()

    start_time = time.time()

    # 画像読み込み グレースケール変換
    img = cv2.imread(sys.argv[1])
    gray_img = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)

    result = compare('Dense','BRISK','BruteForce-Hamming',gray_img)

    end_time = time.time()
    result_time = print_time(end_time - start_time)
