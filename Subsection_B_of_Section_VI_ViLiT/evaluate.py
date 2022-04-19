import os
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch
from torchvision import transforms
from copy import deepcopy
from Subsection_B_of_Section_VI_ViLiT.utils import load_dataset
import shutil
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    transforms = torch.load(
        "C:\\Users\\Razer\\LightDigitDataset\\Blocked Dataset\\Dataset\\model\\transformer_4_block_8_head_scenario2_300.pth")
    transforms = transforms.to(device)
    transforms.eval()
    dataset_size = X_test.shape[0]

    print("Test set size: " + str(dataset_size))
    num_batch = X_test.shape[0] // batch_size + 1
    X = np.array_split(X_test, num_batch)
    y = np.array_split(y_test, num_batch)

    confusion_matrix = torch.zeros((10, 10))
    num_correct = 0
    for inputs, labels in zip(X, y):
        labels = [np.argwhere(e)[0][0] for e in labels]
        inputs = torch.from_numpy(inputs).float()
        labels = torch.Tensor(labels).long()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # inputs = inputs.view([inputs.size()[0], inputs.size()[1], inputs.size()[2] * inputs.size()[3]])
        with torch.no_grad():
            print(inputs.shape)
            outputs = transforms(inputs)
            _, preds = torch.max(outputs, 1)
            num_correct += torch.sum(preds == labels.data)
            for i in range(labels.data.size()[0]):
                confusion_matrix[labels.data[i]][preds[i]] += 1

    print("Number of correct prediction: " + str(num_correct.item()))
    print("Accuracy: " + str(num_correct.item() / dataset_size))
    print(confusion_matrix)