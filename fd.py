#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
import cv2
from optparse import OptionParser
from datetime import datetime
import time
import numpy
import os

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0
counter = 0

def detect_and_draw(img, cascade,c):
    # allocate temporary images
    gray = cv2.cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv2.cv.CreateImage((cv2.cv.Round(img.width / image_scale),
        cv2.cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv2.cv.CvtColor(img, gray, cv2.cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv2.cv.Resize(gray, small_img, cv2.cv.CV_INTER_LINEAR)

    cv2.cv.EqualizeHist(small_img, small_img)

    face_flag = False

    # カスケード分類器による顔検出
    if(cascade):
        t = cv2.cv.GetTickCount()
        faces = cv2.cv.HaarDetectObjects(small_img, cascade, cv2.cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv2.cv.GetTickCount() - t
        print "detection time = %gms" % (t/(cv2.cv.GetTickFrequency()*1000.))
        if faces:
            face_flag = True
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))

                # ある程度顔が検出されたら
                if c > 4:
                    # 画像の保存
                    global counter
                    counter = -1
                    d = datetime.today()
                    datestr = d.strftime('%Y-%m-%d_%H-%M-%S')
                    outputname = '/home/pi/fd/fd_' + datestr + '.jpg'
                    cv2.imwrite(outputname,img)
                    print 'Face Detect'

                    # 読み込みと切り取り
                    fimg = cv2.imread(outputname)
                    fimg_trim = fimg[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    outputname2 = '/home/pi/fd/face_' + datestr + '.jpg'
                    cv2.imwrite(outputname2,fimg_trim)
                    print 'Face Image Save'

                    # 顔判別
                    gray_img = cv2.cvtColor(fimg_trim,cv2.cv.CV_BGR2GRAY)
                    ret = compare('Dense','BRISK','BruteForce-Hamming',gray_img)

                # 顔部分に矩形描画
                cv2.cv.Rectangle(img, pt1, pt2, cv2.cv.RGB(255, 0, 0), 3, 8, 0)

    cv2.cv.ShowImage("result", img)

    return face_flag

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

if __name__ == '__main__':

    # オプション
    parser = OptionParser(usage = "usage: %prog [options] [filename|camera_index]")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, dEfault %default", default = "../data/haarcascades/haarcascade_frontalface_alt.xml")
    #parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "/home/pi/face.xml")
    (options, args) = parser.parse_args()

    cascade = cv2.cv.Load(options.cascade)

    # 引数がなかったら
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    input_name = args[0]
    if input_name.isdigit():
        capture = cv2.cv.CreateCameraCapture(int(input_name))
    else:
        capture = None

    cv2.cv.NamedWindow("result", 1)

    width = 320 #leave None for auto-detection
    height = 240 #leave None for auto-detection

    if width is None:
        width = int(cv2.cv.GetCaptureProperty(capture, cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    else:
        cv2.cv.SetCaptureProperty(capture,cv2.cv.CV_CAP_PROP_FRAME_WIDTH,width)    

    if height is None:
        height = int(cv2.cv.GetCaptureProperty(capture, cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    else:
        cv2.cv.SetCaptureProperty(capture,cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,height) 

    if capture:
        frame_copy = None
        while True:
            detect_flag = False

            frame = cv2.cv.QueryFrame(capture)
            if not frame:
                cv2.cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv2.cv.CreateImage((frame.width,frame.height),
                                            cv2.cv.IPL_DEPTH_8U, frame.nChannels)

#                frame_copy = cv.CreateImage((frame.width,frame.height),
#                                            cv.IPL_DEPTH_8U, frame.nChannels)

            if frame.origin == cv2.cv.IPL_ORIGIN_TL:
                cv2.cv.Copy(frame, frame_copy)
            else:
                cv2.cv.Flip(frame, frame_copy, 0)

            detect_flag = detect_and_draw(frame_copy, cascade,counter)

            if detect_flag:
                if counter == -1:
                    print 'Wait a little time...'
                    if cv2.cv.WaitKey(5000) >= 0:
                        break
                counter += 1
                print 'counter:%d' % (counter)
            else:
                counter = 0


            if cv2.cv.WaitKey(10) >= 0:
                break
    else:
        image = cv2.cv.LoadImage(input_name, 1)
        detect_and_draw(image, cascade)
        cv2.cv.WaitKey(0)

    cv2.cv.DestroyWindow("result")

