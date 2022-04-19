import torch
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from sklearn import preprocessing

from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 1

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    [times, rows, clos] = X_train.shape
    transforms = torch.load(
        "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth").to(device)
    # occlusion = Occlusion(transforms)
    # for j in range(10):
    #     attributions_occ = occlusion.attribute(X_train[j],
    #                                            strides=(3, 50, 50),
    #                                            target=,
    #                                            sliding_window_shapes=(3, 60, 60),
    #                                            baselines=0)
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    Y = np.array_split(y_test, num_batch)
    for index in range(2):
        x_tensor = torch.from_numpy(X[index]).float()
        x_tensor = x_tensor.to(device)
        with torch.no_grad():
            result = transforms(x_tensor)
            _, preds = torch.max(result, 1)
        integrated_gradients = IntegratedGradients(transforms)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        Grad = noise_tunnel
        input = x_tensor
        sequence_length = 300
        input_size = 9
        TestingLabel = preds
        hasBaseline = None
        hasFeatureMask = None
        hasSliding_window_shapes = None
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
                                                                target=TestingLabel).data.cpu().numpsy()

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