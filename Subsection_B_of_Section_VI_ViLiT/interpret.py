import torch
import cv2
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from torch.autograd import Variable
from tqdm import tqdm
from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
from captum.attr import Occlusion
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from Subsection_B_of_Section_VI_ViLiT.perturbation import video_perturbation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 1
num_devices = 1


def getTwoStepRescaling(Grad, input, sequence_length, input_size, TestingLabel, hasBaseline=None, hasFeatureMask=None,
                        hasSliding_window_shapes=None):
    input = input.to(device)
    TestingLabel = TestingLabel.to(device)
    assignment = input[0, 0, 0]
    timeGrad = np.zeros((1, sequence_length))
    inputGrad = np.zeros((input_size, 1))
    newGrad = np.zeros((input_size, sequence_length))
    if (hasBaseline == None):
        ActualGrad = Grad.attribute(input, target=TestingLabel).data.cpu().numpy()
    else:
        if (hasFeatureMask != None):
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel,
                                        feature_mask=hasFeatureMask).data.cpu().numpy()
        elif (hasSliding_window_shapes != None):
            ActualGrad = Grad.attribute(input, sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline,
                                        target=TestingLabel).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()

    #     for t in range(sequence_length):
    #         timeGrad[:,t] = np.mean(np.absolute(ActualGrad[0,:,t]))

    for t in range(sequence_length):
        newInput = input.clone()
        newInput[:, :, t] = assignment

        if (hasBaseline == None):
            timeGrad_perTime = Grad.attribute(newInput, target=TestingLabel).data.cpu().numpy()
        else:
            if (hasFeatureMask != None):
                timeGrad_perTime = Grad.attribute(newInput, baselines=hasBaseline, target=TestingLabel,
                                                  feature_mask=hasFeatureMask).data.cpu().numpy()
            elif (hasSliding_window_shapes != None):
                timeGrad_perTime = Grad.attribute(newInput, sliding_window_shapes=hasSliding_window_shapes,
                                                  baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
            else:
                timeGrad_perTime = Grad.attribute(newInput, baselines=hasBaseline,
                                                  target=TestingLabel).data.cpu().numpy()

        timeGrad_perTime = np.absolute(ActualGrad - timeGrad_perTime)
        timeGrad[:, t] = np.sum(timeGrad_perTime)

    timeContibution = preprocessing.minmax_scale(timeGrad, axis=1)
    meanTime = np.quantile(timeContibution, .55)

    for t in range(sequence_length):
        if (timeContibution[0, t] > meanTime):
            for c in range(input_size):
                newInput = input.clone()
                newInput[:, c, t] = assignment

                if (hasBaseline == None):
                    inputGrad_perInput = Grad.attribute(newInput, target=TestingLabel).data.cpu().numpy()
                else:
                    if (hasFeatureMask != None):
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline, target=TestingLabel,
                                                            feature_mask=hasFeatureMask).data.cpu().numpy()
                    elif (hasSliding_window_shapes != None):
                        inputGrad_perInput = Grad.attribute(newInput, sliding_window_shapes=hasSliding_window_shapes,
                                                            baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()
                    else:
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()

                inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
                inputGrad[c, :] = np.sum(inputGrad_perInput)
                # print(t,c,np.sum(inputGrad_perInput),np.sum(input.data.cpu().numpy()))
            # featureContibution=inputGrad
            featureContibution = preprocessing.minmax_scale(inputGrad, axis=0)
        else:
            featureContibution = np.ones((input_size, 1)) * 0.1

        # meanFeature=np.mean(featureContibution, axis=0)
        # for c in range(input_size):
        #     if(featureContibution[c,0]<=meanFeature):
        #         featureContibution[c,0]=0
        for c in range(input_size):
            newGrad[c, t] = timeContibution[0, t] * featureContibution[c, 0]
            # if(newGrad [c,t]==0):
            #  print(timeContibution[0,t],featureContibution[c,0])
    return newGrad


def givenAttGetRescaledSaliency(attributions, isTensor=True):
    if (isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    saliency = saliency.reshape(-1, 200 * 9)
    rescaledSaliency = minmax_scale(saliency, axis=1)
    rescaledSaliency = rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    [times, rows, clos] = X_train.shape
    transforms = torch.load(
        "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth")
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    Y = np.array_split(y_test, num_batch)
    # x = torch.from_numpy(X_test).float().to(device)
    # y = torch.from_numpy(y_test).float().to(device)
    for index in range(200, 201):
        label = 0
        input_x = np.empty([1, 3, 300, 3, 3], dtype=float)
        x_tensor = torch.from_numpy(X[index]).float()
        y_tensor = torch.from_numpy(Y[index]).float()
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        baseline_single = torch.Tensor(np.random.random(x_tensor.shape)).to(device)
        baseline_multiple = torch.Tensor(
            np.random.random((x_tensor.shape[0] * 5, x_tensor.shape[1], x_tensor.shape[2]))).to(
            device)
        timeMask = np.zeros((300, 9), dtype=int)
        for i in range(300):
            timeMask[i, :] = i
        inputMask = np.zeros(x_tensor.shape)
        print(inputMask.shape)
        inputMask[:, :, :] = timeMask
        inputMask = torch.Tensor(inputMask).to(device)
        mask_single = torch.Tensor(timeMask).to(device)
        mask_single = mask_single.reshape(1, 300, 9).to(device)
        with torch.no_grad():
            result = transforms(x_tensor)
            _, preds = torch.max(result, 1)
        # attributions_OS = OS.attribute(x_tensor, sliding_window_shapes=(1, 1), target=preds, baselines=baseline_single)
        # attributions_OS_np = attributions_OS.cpu().numpy()
        # TSR_attributions_OS = getTwoStepRescaling(OS, x_tensor, 300, 9, preds,
        #                                           hasBaseline=baseline_single,
        #                                           hasSliding_window_shapes=(1, 1))
        # TSR_saliency_OS = givenAttGetRescaledSaliency(TSR_attributions_OS, isTensor=False)
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for j in range(batch_size):
            for index1 in Y[index][j]:
                if index1 != 1.0:
                    label = label + 1
                else:
                    break
            # videoName = "/content/gdrive/MyDrive/Dataset/" + str(index) + "_" + "{:.0f}".format(
            #     preds[0].data) + "_" + str(
            #     label) + ".mp4"
            # videoWrite = cv2.VideoWriter(videoName, fourcc, 20.0, (300, 300))
            for i in range(rows):
                temp = X[index][j][i] * (255 / 100)
                x = np.array([
                    [temp[1], temp[2], temp[3]],
                    [temp[7], temp[8], temp[0]],
                    [temp[6], temp[5], temp[4]]
                ])
                input_x[0, 0, i] = x
                input_x[0, 1, i] = x
                input_x[0, 2, i] = x
        temp_y = np.array([label])
        x = torch.from_numpy(input_x).float().to(device)
        y = torch.from_numpy(temp_y).int().to(device)
        sigma = 11 if x.shape[-1] == 112 else 23
        res = video_perturbation(
            transforms, x, y, result, method="step", areas=[0.1],
            sigma=sigma, max_iter=2000, variant="preserve",
            num_devices=num_devices, print_iter=200, perturb_type="blur")[0]
        resArray = res.cpu().numpy()
        index2 = 0
        videoName8 = "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\Subsection_B_of_Section_VI_ViLiT" + \
                     str(index) + "_" + "{:.0f}".format(preds[0].data) + "_" + str(
            label) + "_STEP.mp4"
        videoWrite8 = cv2.VideoWriter(videoName8, 0x00000021, 20.0, (300, 300))
        for pixel in resArray[0, 0, :]:
            temp = X[index][0][index2] * (255 / 100)
            index2 = index2 + 1
            x = np.array([
                    [temp[1], temp[2], temp[3]],
                    [temp[7], temp[8], temp[0]],
                    [temp[6], temp[5], temp[4]]
                ])
            x = x.astype(np.uint8)
            resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            if pixel[1, 2] > 0.4:
                resized[1, 2, 0] = 0
                resized[1, 2, 1] = 0
            if pixel[0, 0] > 0.4:
                resized[0, 0, 0] = 0
                resized[0, 0, 1] = 0
            if pixel[0, 1] > 0.4:
                resized[0, 1, 0] = 0
                resized[0, 1, 1] = 0
            if pixel[0, 2] > 0.4:
                resized[0, 2, 0] = 0
                resized[0, 2, 1] = 0
            if pixel[2, 2] > 0.4:
                resized[2, 2, 0] = 0
                resized[2, 2, 1] = 0
            if pixel[2, 1] > 0.4:
                resized[2, 1, 0] = 0
                resized[2, 1, 1] = 0
            if pixel[2, 0] > 0.4:
                resized[2, 0, 0] = 0
                resized[2, 0, 1] = 0
            if pixel[1, 0] > 0.4:
                resized[1, 0, 0] = 0
                resized[1, 0, 1] = 0
            if pixel[1, 1] > 0.4:
                resized[1, 1, 0] = 0
                resized[1, 1, 1] = 0
            resized = np.kron(resized, np.ones((100, 100, 1)))
            resized = resized.astype(np.uint8)
            videoWrite8.write(resized)
        # x = x.astype(np.uint8)
        # resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        # resized = np.kron(resized, np.ones((100, 100, 1)))
        # resized = resized.astype(np.uint8)
        # videoWrite.write(resized)
        # videoName8 = "/content/gdrive/MyDrive/Dataset/" + str(index) + "_" + "{:.0f}".format(
        #     preds[0].data) + "_" + str(
        #     label) + "_OS.mp4"
        # videoWrite8 = cv2.VideoWriter(videoName8, fourcc, 20.0, (300, 300))
        # for i in range(rows):
        #     temp = X[index][j][i] * (255 / 100)
        #     x = np.array([
        #         [temp[1], temp[2], temp[3]],
        #         [temp[7], temp[8], temp[0]],
        #         [temp[6], temp[5], temp[4]]
        #     ])
        #     x = x.astype(np.uint8)
        #     resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        #     if TSR_saliency_OS[i][0] > 0.4:
        #         resized[1, 2, 0] = 0
        #         resized[1, 2, 1] = 0
        #     if TSR_saliency_OS[i][1] > 0.4:
        #         resized[0, 0, 0] = 0
        #         resized[0, 0, 1] = 0
        #     if TSR_saliency_OS[i][2] > 0.4:
        #         resized[0, 1, 0] = 0
        #         resized[0, 1, 1] = 0
        #     if TSR_saliency_OS[i][3] > 0.4:
        #         resized[0, 2, 0] = 0
        #         resized[0, 2, 1] = 0
        #     if TSR_saliency_OS[i][4] > 0.4:
        #         resized[2, 2, 0] = 0
        #         resized[2, 2, 1] = 0
        #     if TSR_saliency_OS[i][5] > 0.4:
        #         resized[2, 1, 0] = 0
        #         resized[2, 1, 1] = 0
        #     if TSR_saliency_OS[i][6] > 0.4:
        #         resized[2, 0, 0] = 0
        #         resized[2, 0, 1] = 0
        #     if TSR_saliency_OS[i][7] > 0.4:
        #         resized[1, 0, 0] = 0
        #         resized[1, 0, 1] = 0
        #     if TSR_saliency_OS[i][8] > 0.4:
        #         resized[1, 1, 0] = 0
        #         resized[1, 1, 1] = 0
        #     resized = np.kron(resized, np.ones((100, 100, 1)))
        #     resized = resized.astype(np.uint8)
        #     videoWrite8.write(resized)
