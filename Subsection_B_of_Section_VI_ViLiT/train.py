from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import copy
from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
import numpy as np
from Subsection_B_of_Section_VI_ViLiT.transformer_model import Transformer
from captum.attr import Saliency
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, num_epochs=30, batch_size=1):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                X = X_train
                y = y_train
                model.train()
            else:
                X = X_val
                y = y_val
                model.eval()
            dataset_size = X.shape[0]
            loss = 0.0
            num_correct = 0
            num_batch = X.shape[0] // batch_size + 1
            X = np.array_split(X, num_batch)
            y = np.array_split(y, num_batch)

            for inputs, labels in zip(X, y):
                labels = [np.argwhere(e)[0][0] for e in labels]
                inputs = torch.from_numpy(inputs).float()
                labels = torch.Tensor(labels).long()
                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs = inputs.view([inputs.size()[0], inputs.size()[1], inputs.size()[2] * inputs.size()[3]])
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    current_loss = criterion(outputs, labels)
                    if phase == 'train':
                        current_loss.backward()
                        optimizer.step()
                # print(current_loss.item())
                loss += current_loss.item() * inputs.size(0)
                num_correct += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = loss / dataset_size
            epoch_acc = num_correct.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Store the best model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load the best model and return
    model.load_state_dict(best_model_wts)
    return model


def evaluate(model, X_test, y_test):
    batch_size = 1
    model.eval()
    dataset_size = X_test.shape[0]
    confusion_matrix = torch.zeros((10, 10))
    print("Test set size: " + str(dataset_size))
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    y = np.array_split(y_test, num_batch)

    num_correct = 0
    for inputs, labels in zip(X, y):
        labels = [np.argwhere(e)[0][0] for e in labels]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.Tensor(labels).long()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # inputs = inputs.view([inputs.size()[0], inputs.size()[1], inputs.size()[2] * inputs.size()[3]])

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            num_correct += torch.sum(preds == labels.data)
            for i in range(labels.data.size()[0]):
                confusion_matrix[labels.data[i]][preds[i]] += 1

    print("Number of correct prediction: " + str(num_correct.item()))
    print("Accuracy: " + str(num_correct.item() / dataset_size))
    print(confusion_matrix)


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
    transformer = Transformer(num_block=4)
    transformer = transformer.to(device)
    bce_criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    adam_optimizer = optim.Adam(transformer.parameters())

    # Decay LR by a factor of 0.1 every 10 epochs
    lr_scheduler = lr_scheduler.StepLR(adam_optimizer, step_size=5, gamma=0.1)
    transformer = train(transformer, bce_criterion, adam_optimizer, lr_scheduler, X_train, y_train, X_val, y_val,
                        num_epochs=30)
    torch.save(transformer,
               "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth")
    # transformer = torch.load(
    #     "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth")
    evaluate(transformer, X_test, y_test)
    # transformer = transformer.to(device)
    # ### --------------- interpretability ------------------------##
    # Grad = Saliency(transformer)
    # x_tensor = Variable(torch.from_numpy(X_train).float(), volatile=False, requires_grad=True)
    # y_tensor = Variable(torch.from_numpy(y_train).float(), volatile=False, requires_grad=True)
    # x_tensor = x_tensor.to(device)
    # y_tensor = y_tensor.to(device)
    # TSR_attributions = getTwoStepRescaling(Grad, x_tensor, 9, 200, y_tensor, hasBaseline=None)
    # TSR_saliency = givenAttGetRescaledSaliency(TSR_attributions, isTensor=True)
    # print(TSR_saliency)


