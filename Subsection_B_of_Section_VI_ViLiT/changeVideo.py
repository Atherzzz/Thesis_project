import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, NoiseTunnel
from sklearn import preprocessing

if __name__ == '__main__':
    cap = cv2.VideoCapture('C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\video\\100_3_3.mp4')
    wid = int(cap.get(3))
    hei = int(cap.get(4))
    frameNum = int(cap.get(7))
    video = np.zeros((frameNum, hei, wid, 3), dtype='float16')
    videoWrite = cv2.VideoWriter("C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\video\\goldenAnswer.mp4",
                                 0x00000021, 20.0, (300, 300))
    cap2 = cv2.VideoCapture('C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\video\\100_3_3_OS.mp4')
    saliency = np.zeros((frameNum, hei, wid, 3), dtype='float16')
    res = [False, False, False, False, False, False, False, False, False]
    for i in range(frameNum):
        a, b = cap.read()
        a1, b1 = cap2.read()
        if 55 <= i <= 75:
            if b1[0, 0, 0] < 10:
                res[0] = True
            b[0:100, 0:100, 0] = 0
            b[0:100, 0:100, 1] = 0
        if 75 <= i <= 95:
            if b1[0, 100, 0] < 20:
                res[1] = True
            b[0:100, 100:200, 0] = 0
            b[0:100, 100:200, 1] = 0
        if 95 <= i <= 115:
            if b1[0, 200, 0] < 20:
                res[2] = True
            b[0:100, 200:300, 0] = 0
            b[0:100, 200:300, 1] = 0
        if 115 <= i <= 135:
            if b1[100, 200, 0] < 20:
                res[3] = True
            b[100:200, 200:300, 0] = 0
            b[100:200, 200:300, 1] = 0
        if 135 <= i <= 195:
            if b1[100, 100, 0] < 20:
                res[4] = True
            b[100:200, 100:200, 0] = 0
            b[100:200, 100:200, 1] = 0
        if 195 <= i <= 215:
            if b1[100, 200, 0] < 20:
                res[5] = True
            b[100:200, 200:300, 0] = 0
            b[100:200, 200:300, 1] = 0
        if 215 <= i <= 235:
            if b1[200, 200, 0] < 20:
                res[6] = True
            b[200:300, 200:300, 0] = 0
            b[200:300, 200:300, 1] = 0
        if 235 <= i <= 255:
            if b1[200, 100, 0] < 20:
                res[7] = True
            b[200:300, 100:200, 0] = 0
            b[200:300, 100:200, 1] = 0
        if 255 <= i <= 275:
            if b1[200, 0, 0] < 20:
                res[8] = True
            b[200:300, 0:100, 0] = 0
            b[200:300, 0:100, 1] = 0
        video[i] = b
        saliency[i] = b1
        frame = b.astype(np.uint8)
        videoWrite.write(frame)
    print("finished")
    score = 0
    for temp in res:
        if temp:
            score = score + 1
    print(score)