import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, NoiseTunnel
from sklearn import preprocessing

from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 1

if __name__ == '__main__':
    X = np.arange(1, 256, 5)
    Score = []
    relativeScore = []
    coverScore = []
    for x in X:
        correctNum = 0
        amount = 0
        appearRes = {}
        corrected = {"7------>5": False, "5------>7": False, "6------>5": False, "6------>9": False, "6------>8": False,
                     "6------>3": False, "3------>2": False, "5------>2": False}
        appearNum = 3
        dif = 10
        thresh = x
        lastFrame = 1
        frameDiff = 20
        cap = cv2.VideoCapture('C:\\Users\\Razer\\Desktop\\video\\TestData\\0\\0_8_0_OS.mp4')
        wid = int(cap.get(3))
        hei = int(cap.get(4))
        frameNum = int(cap.get(7))
        video = np.zeros((frameNum, hei, wid, 3), dtype='float16')
        cnt = 0
        frameList = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        timeList = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        printList = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False}
        for i in range(frameNum):
            a, b = cap.read()
            video[i] = b
            if abs(int(b[0][0][0]) - int(b[0][0][2])) > dif and b[0][0][2] < thresh:
                timeList[1] = timeList[1] + 1
                if timeList[1] >= lastFrame and not printList[1]:
                    # print(str(cnt) + " 1")
                    frameList[1].append(cnt)
                    printList[1] = True
            else:
                timeList[1] = 0
                printList[1] = False
            if abs(int(b[0][100][0]) - int(b[0][100][2])) > dif and b[0][100][2] < thresh:
                # if i == 126:
                #     print("hh")
                # if i == 127:
                #     print("hh")
                timeList[2] = timeList[2] + 1
                if timeList[2] >= lastFrame and not printList[2]:
                    # print(str(cnt) + " 2")
                    frameList[2].append(cnt)
                    printList[2] = True
            else:
                timeList[2] = 0
                printList[2] = False
            if abs(int(b[0][200][0]) - int(b[0][200][2])) > dif and b[0][200][2] < thresh:
                timeList[3] = timeList[3] + 1
                if timeList[3] >= lastFrame and not printList[3]:
                    # print(str(cnt) + " 3")
                    frameList[3].append(cnt)
                    printList[3] = True
            else:
                timeList[3] = 0
                printList[3] = False
            if abs(int(b[100][0][0]) - int(b[100][0][2])) > dif and b[100][0][2] < thresh:
                timeList[4] = timeList[4] + 1
                if timeList[4] >= lastFrame and not printList[4]:
                    # print(str(cnt) + " 4")
                    frameList[4].append(cnt)
                    printList[4] = True
            else:
                timeList[4] = 0
                printList[4] = False
            if abs(int(b[100][100][0]) - int(b[100][100][2])) > dif and b[100][100][2] < thresh:
                timeList[5] = timeList[5] + 1
                if timeList[5] >= lastFrame and not printList[5]:
                    # print(str(cnt) + " 5")
                    frameList[5].append(cnt)
                    printList[5] = True
            else:
                timeList[5] = 0
                printList[5] = False
            if abs(int(b[100][200][0]) - int(b[100][200][2])) > dif and b[100][200][2] < thresh:
                timeList[6] = timeList[6] + 1
                if timeList[6] >= lastFrame and not printList[6]:
                    # print(str(cnt) + " 6")
                    frameList[6].append(cnt)
                    printList[6] = True
            else:
                timeList[6] = 0
                printList[6] = False
            if abs(int(b[200][0][0]) - int(b[200][0][2])) > dif and b[200][0][2] < thresh:
                # print(str(cnt))
                timeList[7] = timeList[7] + 1
                if timeList[7] >= lastFrame and not printList[7]:
                    # print(str(cnt) + " 7")
                    frameList[7].append(cnt)
                    printList[7] = True
            else:
                timeList[7] = 0
                printList[7] = False
            if abs(int(b[200][100][0]) - int(b[200][100][2])) > dif and b[200][100][2] < thresh:
                timeList[8] = timeList[8] + 1
                if timeList[8] >= lastFrame and not printList[8]:
                    # print(str(cnt) + " 8")
                    frameList[8].append(cnt)
                    printList[8] = True
            else:
                timeList[8] = 0
                printList[8] = False
            if abs(int(b[200][200][0]) - int(b[200][200][2])) > dif and b[200][200][2] < thresh:
                timeList[9] = timeList[9] + 1
                if timeList[9] >= lastFrame and not printList[9]:
                    # print(str(cnt) + " 9")
                    frameList[9].append(cnt)
                    printList[9] = True
            else:
                timeList[9] = 0
                printList[9] = False
            cnt = cnt + 1
        temp = frameList.copy()
        for key in frameList.keys():
            if not frameList.get(key):
                continue
            for secondKey in frameList.keys():
                if secondKey == key or not frameList.get(secondKey):
                    continue
                for value in frameList[key]:
                    for secondValue in frameList[secondKey]:
                        if frameDiff >= secondValue - value > 0:
                            # print(str(key) + "------>" + str(secondKey) + ":" + str(value) + "----------->" + str(secondValue))
                            if (str(key) + "------>" + str(secondKey)) in appearRes.keys():
                                appearRes[str(key) + "------>" + str(secondKey)] = appearRes[str(key) + "------>" + str(
                                    secondKey)] + 1
                            else:
                                appearRes[str(key) + "------>" + str(secondKey)] = 1
        realRes = {}
        for key in appearRes.keys():
            if appearRes[key] >= appearNum:
                realRes[key] = appearRes[key]
        for key in realRes.keys():
            amount = amount + realRes[key]
            if key in corrected:
                corrected[key] = True
                correctNum = correctNum + realRes[key]
        counter = 0
        for value in corrected.values():
            if value:
                counter = counter + 1
        Score.append(counter / len(corrected))
        if amount == 0:
            relativeScore.append(1)
        else:
            relativeScore.append(correctNum / amount)
        coverScore.append(1-len(realRes)/72)
    scoreArray = np.array(Score)
    relativeScoreArray = np.array(relativeScore)
    coverScoreArray = np.array(coverScore)
    plt.plot(X, scoreArray, label="Score")
    plt.plot(X, relativeScoreArray, label="relativeScore")
    plt.plot(X, coverScoreArray, label="coverScore")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("parameter------threshold")
    plt.legend()
    plt.show()