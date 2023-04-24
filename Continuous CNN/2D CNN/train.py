import numpy as np
from scipy import signal
from dataset import Dataset
from model import Model_name,Model_name_img
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score,precision_score,recall_score

# stroke probability map
def stroke_probability_map_1(data, model):
    split_list = [125, 200, 400]
    probability_map = np.zeros((16, 12*len(split_list)))
    N = data.shape[0]
    count = 0
    segments = 16
    for split_idx in range(len(split_list)):
        step = int(np.floor((N-split_list[split_idx])/(segments-1)))
        data_cur = np.zeros((segments, split_list[split_idx], 2))
        for s_idx in range(segments):
            data_cur[s_idx] = data[(s_idx*step):(s_idx*step+split_list[split_idx]),:]
        data_cur = signal.resample(data_cur, 100, axis=1)
        data_cur = data_cur.reshape((segments, -1))
        probability_map[:, (split_idx*12):(split_idx*12+12)] = model.predict_proba(data_cur)
            
    return probability_map


# load data loder
def split_data(test_split,val_split):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    train_stroke = []
    train_img = []
    val_stroke = []
    val_img = []
    test_stroke = []
    test_img = []
    patient = ["001","002","003","004","005","006"]
    for i in os.listdir("/data/achar15/MLMA/continuous/"):
        for p in patient:
            if i.endswith("cvs"):
                data = np.array(pd.read_csv(str("/data/achar15/MLMA/continuous/"+p+"/training/"+i),names=["vertical","horizontal"]))
                data = signal.resample(data, 1024, axis=1)
                with open(str("../../Isolated_model/SVC__sub"+p+".pck"), "rb") as input_file:
                    model = pickle.load(input_file)
                data = stroke_probability_map_1(data, model)
                if i.split("_")[2] == test_split:
                    test_stroke.append(data)
                elif i.split("_")[2] == val_split:
                    val_stroke.append(data)
                else:
                    train_stroke.append(data)
            elif i.endswith("png"):
                img = str("/data/achar15/MLMA/continuous/"+p+"/training/"+i)
                if i.split("_")[2] == test_split:
                    test_img.append(img)
                elif i.split("_")[2] == val_split:
                    val_img.append(img)
                else:
                    train_img.append(img)
    train_data = Dataset(train_stroke,train_img,transform=transform)
    test_data = Dataset(test_stroke,test_img,transform=transform)
    val_data =Dataset(val_stroke,val_img,transform=transform)
    train_loader = data_utils.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = data_utils.DataLoader(val_data, batch_size=32, shuffle=True)
    test_loader = data_utils.DataLoader(test_data, batch_size=32, shuffle=True)
    return train_loader,val_loader,test_loader


def evaluate(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred,average = 'micro')
    f1_macro = f1_score(y_true, y_pred,average = 'macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro,acc

def validate(model, val_loader,criterion,device):
    model.eval()
    val_loss = 0
    ytrue = []
    ypred = []
    # .....
    with torch.no_grad():
        for i, data in enumerate(val_loader,0):
            data,img,label= data
            data,img,label= data.to(device), img.to(device),label.to(device)
            ytrue.extend(label.tolist())
            outputs = model(data,img)
            val_loss += criterion(outputs, label)
            pred = torch.argmax(outputs)
            ypred.extend(pred.tolist())

    f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro,acc = evaluate(ytrue, ypred)
            
    return val_loss/(i+1),f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro,acc

def train(model, train_loader,val_loader,test_loader,device,criterion,optimizer,num_epochs):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        print("running epoch",epoch)
        loss_train = 0.0
        ytrue = []
        ypred = []
        for i, data in enumerate(train_loader, 0): # each time, load a batch of data from data loader
            data,img,label= data
            data,img,label= data.to(device), img.to(device),label.to(device)
            ytrue.extend(label.tolist())
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs= model(data,img)
            loss = criterion(outputs, label)
            # print("batch",i, loss)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = torch.argmax(outputs)
            ypred.extend(pred.tolist())
        f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro,accuracy_train = evaluate(ytrue, ypred)

        loss_train = loss_train/(i+1) # calculate the avergaed loss over all batches
        loss_val, f1_micro,f1_macro,precision_micro,precision_macro,recall_micro,recall_macro,accuracy_val = validate(model, val_loader,criterion,device)
        # let's print these loss out
        print("epoch {}, training loss = {:.3f}, validation loss = {:.3f}, training accuracy = {:.3f}, validation accuracy = {:.3f}".format(
            epoch,
            loss_train,
            loss_val,
            accuracy_train,
            accuracy_val 
        ))
    return model






