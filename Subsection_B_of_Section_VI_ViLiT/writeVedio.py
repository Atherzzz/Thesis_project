import cv2
import os
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients, NoiseTunnel, GradientShap

from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
from captum.attr import visualization as viz
import torch

device = torch.device("cpu")
batch_size = 16
#cv2.VideoWriter_fourcc(*'MJPG')
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    [times, rows, clos] = X_train.shape
    transforms = torch.load(
        "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth")
    # occlusion = Occlusion(transforms)
    transforms.to(device)
    # for j in range(10):
    #     attributions_occ = occlusion.attribute(X_train[j],
    #                                            strides=(3, 50, 50),
    #                                            target=,
    #                                            sliding_window_shapes=(3, 60, 60),
    #                                            baselines=0)
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    Y = np.array_split(y_test, num_batch)
    x_tensor = torch.from_numpy(X[0]).float()
    x_tensor = x_tensor.to(device)
    with torch.no_grad():
        result = transforms(x_tensor)
        _, preds = torch.max(result, 1)
    integrated_gradients = IntegratedGradients(transforms)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(x_tensor, nt_samples=5, nt_type='smoothgrad_sq', target=preds)
    attributions_ig_nt_np = attributions_ig_nt.numpy()
    torch.manual_seed(0)
    np.random.seed(0)
    gradient_shap = GradientShap(transforms)
    rand_img_dist = torch.cat([x_tensor * 0, x_tensor * 1])
    attributions_gs = gradient_shap.attribute(x_tensor,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=preds)
    attributions_gs_np = attributions_gs.numpy()
    for j in range(batch_size):
        label = 0
        for index in Y[0][j]:
            if index != 1.0:
                label = label + 1
            else:
                break
        videoWrite = cv2.VideoWriter(
            "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(j) + "_" + str(label) + ".mp4", 0x00000021
            , 20.0, (300, 300))
        for i in range(rows):
            temp = X[0][j][i] * (255 / 100)
            # if temp[1] <= 220:
            #     temp[7] = 255
            #     temp[6] = 255
            # if temp[7] <= 220:
            #     temp[6] = 255
            # if temp[2] <= 220:
            #     temp[8] = 255
            #     temp[5] = 255
            # if temp[8] <= 220:
            #     temp[5] = 255
            # if temp[3] <= 220:
            #     temp[0] = 255
            #     temp[4] = 255
            # if temp[0] <= 220:
            #     temp[4] = 255
            for index in range(9):
                if temp[index] > 220:
                    temp[index] = 255
            x = np.array([
                [temp[1], temp[2], temp[3]],
                [temp[7], temp[8], temp[0]],
                [temp[6], temp[5], temp[4]]
            ])
            x = x.astype(np.uint8)
            resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            if attributions_gs_np[j][i][0] > 10:
                resized[1, 2, 0] = 0
                resized[1, 2, 1] = 0
            if attributions_gs_np[j][i][1] > 10:
                resized[0, 0, 0] = 0
                resized[0, 0, 1] = 0
            if attributions_gs_np[j][i][2] > 10:
                resized[0, 1, 0] = 0
                resized[0, 1, 1] = 0
            if attributions_gs_np[j][i][3] > 10:
                resized[0, 2, 0] = 0
                resized[0, 2, 1] = 0
            if attributions_gs_np[j][i][4] > 10:
                resized[2, 2, 0] = 0
                resized[2, 2, 1] = 0
            if attributions_gs_np[j][i][5] > 10:
                resized[2, 1, 0] = 0
                resized[2, 1, 1] = 0
            if attributions_gs_np[j][i][6] > 10:
                resized[2, 0, 0] = 0
                resized[2, 0, 1] = 0
            if attributions_gs_np[j][i][7] > 10:
                resized[1, 0, 0] = 0
                resized[1, 0, 1] = 0
            if attributions_gs_np[j][i][8] > 10:
                resized[1, 1, 0] = 0
                resized[1, 1, 1] = 0
            resized = np.kron(resized, np.ones((100, 100, 1)))
            resized = resized.astype(np.uint8)
            videoWrite.write(resized)
        videoWrite2 = cv2.VideoWriter(
                "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(j) + "_" + str(label) + "_before.mp4",
                0x00000021, 20.0, (300, 300))
        for i in range(rows):
            temp = X[0][j][i] * (255 / 100)
            if temp[1] <= 220:
                temp[7] = 255
                temp[6] = 255
            if temp[7] <= 220:
                temp[6] = 255
            if temp[2] <= 220:
                temp[8] = 255
                temp[5] = 255
            if temp[8] <= 220:
                temp[5] = 255
            if temp[3] <= 220:
                temp[0] = 255
                temp[4] = 255
            if temp[0] <= 220:
                temp[4] = 255
            for index in range(9):
                if temp[index] > 220:
                    temp[index] = 255
            x = np.array([
                [temp[1], temp[2], temp[3]],
                [temp[7], temp[8], temp[0]],
                [temp[6], temp[5], temp[4]]
            ])
            resized = np.kron(x, np.ones((100, 100)))
            resized = resized.astype(np.uint8)
            videoWrite2.write(resized)
        videoWrite3 = cv2.VideoWriter(
                "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(j) + "_" + str(label) + "after.mp4",
                0x00000021, 20.0, (300, 300))
        for i in range(rows):
            temp = X[0][j][i] * (255 / 100)
            if temp[1] <= 220:
                temp[7] = 255
                temp[6] = 255
            if temp[7] <= 220:
                temp[6] = 255
            if temp[2] <= 220:
                temp[8] = 255
                temp[5] = 255
            if temp[8] <= 220:
                temp[5] = 255
            if temp[3] <= 220:
                temp[0] = 255
                temp[4] = 255
            if temp[0] <= 220:
                temp[4] = 255
            for index in range(9):
                if temp[index] > 220:
                    temp[index] = 255
            x = np.array([
                [temp[1], temp[2], temp[3]],
                [temp[7], temp[8], temp[0]],
                [temp[6], temp[5], temp[4]]
            ])
            x = x.astype(np.uint8)
            resized = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            if attributions_ig_nt_np[j][i][0] > 200:
                resized[1, 2, 0] = 0
                resized[1, 2, 1] = 0
            if attributions_ig_nt_np[j][i][1] > 200:
                resized[0, 0, 0] = 0
                resized[0, 0, 1] = 0
            if attributions_ig_nt_np[j][i][2] > 200:
                resized[0, 1, 0] = 0
                resized[0, 1, 1] = 0
            if attributions_ig_nt_np[j][i][3] > 200:
                resized[0, 2, 0] = 0
                resized[0, 2, 1] = 0
            if attributions_ig_nt_np[j][i][4] > 200:
                resized[2, 2, 0] = 0
                resized[2, 2, 1] = 0
            if attributions_ig_nt_np[j][i][5] > 200:
                resized[2, 1, 0] = 0
                resized[2, 1, 1] = 0
            if attributions_ig_nt_np[j][i][6] > 200:
                resized[2, 0, 0] = 0
                resized[2, 0, 1] = 0
            if attributions_ig_nt_np[j][i][7] > 200:
                resized[1, 0, 0] = 0
                resized[1, 0, 1] = 0
            if attributions_ig_nt_np[j][i][8] > 200:
                resized[1, 1, 0] = 0
                resized[1, 1, 1] = 0
            resized = np.kron(resized, np.ones((100, 100, 1)))
            resized = resized.astype(np.uint8)
            videoWrite3.write(resized)
        # videoWrite3 = cv2.VideoWriter(
        #         "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(j) + "_" + str(label) + "after.mp4",
        #         0x00000021, 20.0, (300, 300))
        # for i in range(rows):
        #     temp = X_train[j, i] * (255 / 100)
        #     if temp[1] <= 220:
        #         temp[7] = 220
        #         temp[6] = 220
        #     if temp[7] <= 220:
        #         temp[6] = 220
        #     if temp[2] <= 220:
        #         temp[8] = 220
        #         temp[5] = 220
        #     if temp[8] <= 220:
        #         temp[5] = 220
        #     if temp[3] <= 220:
        #         temp[0] = 220
        #         temp[4] = 220
        #     if temp[0] <= 220:
        #         temp[4] = 220
        #     for index in range(9):
        #         if temp[index] > 220:
        #             temp[index] = 220
        #     x = np.array([
        #         [temp[1], temp[2], temp[3]],
        #         [temp[7], temp[8], temp[0]],
        #         [temp[6], temp[5], temp[4]]
        #     ])
        #     resized = np.kron(x, np.ones((100, 100)))
        #     resized = resized.astype(np.uint8)
        #     videoWrite3.write(resized)
    # for x, y in zip(X[0], Y[0]):
    #     counter = 1
    #     label = 0
    #     for index in y:
    #         if index != 1.0:
    #             label = label + 1
    #         else:
    #             break
    #     videoWrite = cv2.VideoWriter(
    #         "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\" + str(counter) + "_" + str(
    #             label) + ".mp4",
    #         0x00000021, 20.0, (300, 300))
    #     counter = counter + 1
# ----------------------------#

