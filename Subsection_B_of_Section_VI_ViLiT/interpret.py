import torch
import cv2
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from torch.autograd import Variable

from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
from captum.attr import Occlusion
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 2

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


def givenAttGetRescaledSaliency(attributions, isTensor = True):
    if(isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    saliency = saliency.reshape(-1, 200*9)
    rescaledSaliency = minmax_scale(saliency, axis=1)
    rescaledSaliency = rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency

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
    x_tensor = torch.from_numpy(X[46]).float()
    x_tensor = x_tensor.to(device)
    with torch.no_grad():
        result = transforms(x_tensor)
        _, preds = torch.max(result, 1)
    integrated_gradients = IntegratedGradients(transforms)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    x_tensor = x_tensor.to(device)
    preds = preds.to(device)
    TSR_attributions = getTwoStepRescaling(noise_tunnel, x_tensor, 9, 200, preds)
    TSR_saliency = givenAttGetRescaledSaliency(TSR_attributions, isTensor=False)
    print(TSR_saliency)