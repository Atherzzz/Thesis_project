import torch
import cv2
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel, GradientShap, Saliency, DeepLiftShap, DeepLift, \
    ShapleyValueSampling
from torch.autograd import Variable

from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
from captum.attr import Occlusion
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 1


def getTwoStepRescaling(Grad, input, sequence_length, input_size, TestingLabel, hasBaseline=None, hasFeatureMask=None,
                        hasSliding_window_shapes=None):
    input = input.to(device)
    TestingLabel = TestingLabel.to(device)
    assignment = input[0, 0, 0]
    timeGrad = np.zeros((sequence_length, 1))
    inputGrad = np.zeros((1, input_size))
    newGrad = np.zeros((sequence_length, input_size))
    if (hasBaseline == None):
        ActualGrad = Grad.attribute(input, target=TestingLabel).data.cpu().numpy()
    else:
        if (hasFeatureMask != None):
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel,
                                        feature_mask=hasFeatureMask).data.cpu().numpy()
        elif (hasSliding_window_shapes != None):
            ActualGrad = Grad.attribute(input, sliding_window_shapes=hasSliding_window_shapes,
                                        baselines=hasBaseline,
                                        target=TestingLabel).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()

    #     for t in range(sequence_length):
    #         timeGrad[:,t] = np.mean(np.absolute(ActualGrad[0,:,t]))

    for t in range(sequence_length):
        newInput = input.clone()
        newInput[:, t, :] = assignment
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
        timeGrad[t, :] = np.sum(timeGrad_perTime)
    timeContibution = preprocessing.minmax_scale(timeGrad, axis=0)
    meanTime = np.quantile(timeContibution, .55)

    for t in range(sequence_length):
        if (timeContibution[t, 0] > meanTime):
            for c in range(input_size):
                newInput = input.clone()
                newInput[:, t, c] = assignment

                if (hasBaseline == None):
                    inputGrad_perInput = Grad.attribute(newInput, target=TestingLabel).data.cpu().numpy()
                else:
                    if (hasFeatureMask != None):
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline, target=TestingLabel,
                                                            feature_mask=hasFeatureMask).data.cpu().numpy()
                    elif (hasSliding_window_shapes != None):
                        inputGrad_perInput = Grad.attribute(newInput,
                                                            sliding_window_shapes=hasSliding_window_shapes,
                                                            baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()
                    else:
                        inputGrad_perInput = Grad.attribute(newInput, baselines=hasBaseline,
                                                            target=TestingLabel).data.cpu().numpy()

                inputGrad_perInput = np.absolute(ActualGrad - inputGrad_perInput)
                inputGrad[:, c] = np.sum(inputGrad_perInput)
                # print(t,c,np.sum(inputGrad_perInput),np.sum(input.data.cpu().numpy()))
            # featureContibution=inputGrad
            featureContibution = preprocessing.minmax_scale(inputGrad, axis=1)
        else:
            featureContibution = np.ones((1, input_size)) * 0.1

        # meanFeature=np.mean(featureContibution, axis=0)
        # for c in range(input_size):
        #     if(featureContibution[c,0]<=meanFeature):
        #         featureContibution[c,0]=0
        for c in range(input_size):
            newGrad[t, c] = timeContibution[t, 0] * featureContibution[0, c]
            # if(newGrad [c,t]==0):
            #  print(timeContibution[0,t],featureContibution[c,0])
    return newGrad


def givenAttGetRescaledSaliency(attributions, isTensor=True):
    if (isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    saliency = saliency.reshape(-1, 300 * 9)
    rescaledSaliency = minmax_scale(saliency, axis=1)
    rescaledSaliency = rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    [times, rows, clos] = X_train.shape
    transforms = torch.load(
        "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth").to(
        device)
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    Y = np.array_split(y_test, num_batch)
    for index in (100, 200):
        x_tensor = torch.from_numpy(X[index]).float()
        x_tensor = x_tensor.to(device)
        # baseline_single = torch.Tensor(np.random.random(x_tensor.shape)).to(device)
        # baseline_multiple = torch.Tensor(
        #     np.random.random((x_tensor.shape[0] * 5, x_tensor.shape[1], x_tensor.shape[2]))).to(
        #     device)
        # timeMask = np.zeros((300, 9),  dtype=int)
        # for i in range(300):
        #     timeMask[i, :] = i
        # inputMask = np.zeros(x_tensor.shape)
        # inputMask[:, :, :] = timeMask
        # inputMask = torch.Tensor(inputMask).to(device)
        # mask_single = torch.Tensor(timeMask).to(device)
        # mask_single = mask_single.reshape(1, 300, 9).to(device)
        # with torch.no_grad():
        #     result = transforms(x_tensor)
        #     _, preds = torch.max(result, 1)
        # # -------------------Integrated_gradients--------------------------#
        # integrated_gradients = IntegratedGradients(transforms)
        # attributions_ig_nt = integrated_gradients.attribute(x_tensor, baselines=baseline_single, target=preds)
        # attributions_ig_nt_np = attributions_ig_nt.cpu().numpy()
        # x_tensor = x_tensor.to(device)
        # preds = preds.to(device)
        # TSR_attributions_IG = getTwoStepRescaling(integrated_gradients, x_tensor, 300, 9, preds,
        #                                           hasBaseline=baseline_single)
        # TSR_saliency_IG = givenAttGetRescaledSaliency(TSR_attributions_IG, isTensor=False)
        # # --------------------Gradient Shap-----------------------------#
        # torch.manual_seed(0)
        # np.random.seed(0)
        # gradient_shap = GradientShap(transforms)
        # attributions_gs = gradient_shap.attribute(x_tensor,
        #                                           stdevs=0.09,
        #                                           baselines=baseline_multiple,
        #                                           target=preds)
        # attributions_gs_np = attributions_gs.cpu().numpy()
        # TSR_attributions_GS = getTwoStepRescaling(gradient_shap, x_tensor, 300, 9, preds, hasBaseline=baseline_multiple)
        # TSR_saliency_GS = givenAttGetRescaledSaliency(TSR_attributions_GS, isTensor=False)
        # # ----------------Saliency----------------------------------#
        # Grad = Saliency(transforms)
        # attributions_g = Grad.attribute(x_tensor, target=preds)
        # attributions_g_np = attributions_g.cpu().numpy()
        # TSR_attributions_G = getTwoStepRescaling(Grad, x_tensor, 300, 9, preds, hasBaseline=None)
        # TSR_saliency_G = givenAttGetRescaledSaliency(TSR_attributions_G, isTensor=False)
        # # --------------DeepLift----------------------------------#
        # DL = DeepLift(transforms)
        # attributions_dl = DL.attribute(x_tensor, baselines=baseline_single, target=preds)
        # attributions_dl_np = attributions_dl.cpu().detach().numpy()
        # TSR_attributions_dl = getTwoStepRescaling(DL, x_tensor, 300, 9, preds, hasBaseline=baseline_single)
        # TSR_saliency_dl = givenAttGetRescaledSaliency(TSR_attributions_dl, isTensor=False)
        # # -------------DeepLiftShap------------------------------#
        # DLS = DeepLiftShap(transforms)
        # TSR_attributions_dls = getTwoStepRescaling(DLS, x_tensor, 300, 9, preds,
        #                                            hasBaseline=baseline_multiple)
        # TSR_saliency_dls = givenAttGetRescaledSaliency(TSR_attributions_dls, isTensor=False)
        # -------------SharpleyValueSampling----------------------#
        SS = ShapleyValueSampling(transforms)
        attributions_SS = SS.attribute(x_tensor, baselines=baseline_single, target=preds, feature_mask=inputMask)
        attributions_SS_np = attributions_SS.cpu().numpy()
        TSR_attributions_SS = getTwoStepRescaling(SS, x_tensor, 300, 9, preds, hasBaseline=baseline_single,
                                                  hasFeatureMask=inputMask)
        TSR_saliency_SS = givenAttGetRescaledSaliency(TSR_attributions_SS, isTensor=False)
        # ------------OcclusionFlag-----------------------------#
        OS = Occlusion(transforms)
        attributions_OS = OS.attribute(x_tensor, sliding_window_shapes=(1, 1), target=preds, baselines=baseline_single)
        attributions_OS_np = attributions_OS.cpu().numpy()
        TSR_attributions_OS = getTwoStepRescaling(OS, x_tensor, 300, 9, preds,
                                                  hasBaseline=baseline_single,
                                                  hasSliding_window_shapes=(1, 1))
        TSR_saliency_OS = givenAttGetRescaledSaliency(TSR_attributions_OS, isTensor=False)
        for j in range(batch_size):
            label = 0
            for index1 in Y[index][j]:
                if index1 != 1.0:
                    label = label + 1
                else:
                    break
#-------------------------Wirte Video for IG------------------------#
            videoWrite = cv2.VideoWriter(
                "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
                    label) + "_IG.mp4", 0x00000021
                , 20.0, (300, 300))
            for i in range(rows):
                temp = X[index][j][i] * (255 / 100)
                x = np.array([
                    [temp[1], temp[2], temp[3]],
                    [temp[7], temp[8], temp[0]],
                    [temp[6], temp[5], temp[4]]
                ])
                x = x.astype(np.uint8)
                resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
                # if TSR_saliency_IG[i][0] > 0.4:
                #     resized[1, 2, 0] = 0
                #     resized[1, 2, 1] = 0
                # if TSR_saliency_IG[i][1] > 0.4:
                #     resized[0, 0, 0] = 0
                #     resized[0, 0, 1] = 0
                # if TSR_saliency_IG[i][2] > 0.4:
                #     resized[0, 1, 0] = 0
                #     resized[0, 1, 1] = 0
                # if TSR_saliency_IG[i][3] > 0.4:
                #     resized[0, 2, 0] = 0
                #     resized[0, 2, 1] = 0
                # if TSR_saliency_IG[i][4] > 0.4:
                #     resized[2, 2, 0] = 0
                #     resized[2, 2, 1] = 0
                # if TSR_saliency_IG[i][5] > 0.4:
                #     resized[2, 1, 0] = 0
                #     resized[2, 1, 1] = 0
                # if TSR_saliency_IG[i][6] > 0.4:
                #     resized[2, 0, 0] = 0
                #     resized[2, 0, 1] = 0
                # if TSR_saliency_IG[i][7] > 0.4:
                #     resized[1, 0, 0] = 0
                #     resized[1, 0, 1] = 0
                # if TSR_saliency_IG[i][8] > 0.4:
                #     resized[1, 1, 0] = 0
                #     resized[1, 1, 1] = 0
                resized = np.kron(resized, np.ones((100, 100, 1)))
                resized = resized.astype(np.uint8)
                videoWrite.write(resized)
# -------------------------Wirte Video for GS------------------------#
#             videoWrite2 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_GS.mp4", 0x00000021
#                 , 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_GS[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_GS[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_GS[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_GS[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_GS[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_GS[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_GS[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_GS[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_GS[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite2.write(resized)
# #--------------------------write video for Original-----------------------#
#             videoWrite3 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_original" + str(
#                     label) + ".mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite3.write(resized)
# #--------------------write video for gradient-----------------------#
#             videoWrite4 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_G.mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_G[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_G[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_G[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_G[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_G[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_G[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_G[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_G[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_G[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite4.write(resized)
# #---------------------write video for deep lift --------------------------#
#             videoWrite5 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_dl.mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_dl[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_dl[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_dl[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_dl[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_dl[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_dl[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_dl[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_dl[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_dl[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite5.write(resized)
# #-------------------write model for deepliftsharp-----------------------#
#             videoWrite6 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_dls.mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_dls[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_dls[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_dls[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_dls[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_dls[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_dls[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_dls[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_dls[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_dls[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite6.write(resized)
# #----------------------------sharpely value prediction-----------------------#
#             videoWrite7 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_SS.mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_SS[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_SS[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_SS[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_SS[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_SS[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_SS[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_SS[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_SS[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_SS[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite7.write(resized)
# #------------------------------write video fpr OcclusionFlag--------------------#
#             videoWrite8 = cv2.VideoWriter(
#                 "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(index) + "_" + str(
#                     label) + "_OS.mp4",
#                 0x00000021, 20.0, (300, 300))
#             for i in range(rows):
#                 temp = X[index][j][i] * (255 / 100)
#                 x = np.array([
#                     [temp[1], temp[2], temp[3]],
#                     [temp[7], temp[8], temp[0]],
#                     [temp[6], temp[5], temp[4]]
#                 ])
#                 x = x.astype(np.uint8)
#                 resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
#                 if TSR_saliency_OS[i][0] > 0.4:
#                     resized[1, 2, 0] = 0
#                     resized[1, 2, 1] = 0
#                 if TSR_saliency_OS[i][1] > 0.4:
#                     resized[0, 0, 0] = 0
#                     resized[0, 0, 1] = 0
#                 if TSR_saliency_OS[i][2] > 0.4:
#                     resized[0, 1, 0] = 0
#                     resized[0, 1, 1] = 0
#                 if TSR_saliency_OS[i][3] > 0.4:
#                     resized[0, 2, 0] = 0
#                     resized[0, 2, 1] = 0
#                 if TSR_saliency_OS[i][4] > 0.4:
#                     resized[2, 2, 0] = 0
#                     resized[2, 2, 1] = 0
#                 if TSR_saliency_OS[i][5] > 0.4:
#                     resized[2, 1, 0] = 0
#                     resized[2, 1, 1] = 0
#                 if TSR_saliency_OS[i][6] > 0.4:
#                     resized[2, 0, 0] = 0
#                     resized[2, 0, 1] = 0
#                 if TSR_saliency_OS[i][7] > 0.4:
#                     resized[1, 0, 0] = 0
#                     resized[1, 0, 1] = 0
#                 if TSR_saliency_OS[i][8] > 0.4:
#                     resized[1, 1, 0] = 0
#                     resized[1, 1, 1] = 0
#                 resized = np.kron(resized, np.ones((100, 100, 1)))
#                 resized = resized.astype(np.uint8)
#                 videoWrite8.write(resized)