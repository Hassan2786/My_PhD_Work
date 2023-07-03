import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os,sys
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
import copy
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
from sklearn.metrics import classification_report
import time
import pickle
import argparse
import seaborn as sns
import datetime
import os, sys
from matplotlib.pyplot import imshow, imsave
from utils import make_variable



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epoch_num_Encoder = 0
epoch_num_Aligner = 0
learning_rate = 0.00146

input_size1 = 1024 #468  # Input size with 114 sub-carriers for each antenna pair with 4 such pairs
input_size2 = 100
hidden_size1 = 100  # Hidden layer Size
hidden_size2 = 16
output_dim = 50   # Output size for predicting 7 activities
num_layers = 4

dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
#batch_size = 50
image_size = 64

d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
num_epochs_S = 2
num_epochs_T = 2

BATCH_SIZE = batch_size = 10
load_model=False

classes = ('bending', 'falling', 'lie_down', 'running',
           'sit_down', 'stand_up', 'walking')

train_dataset_path="./Dataset_S1_S2_S3/training3/training"
test_dataset_path="./Dataset_S1_S2_S3/training2/training"

mean=[0.7290,0.8188,0.6578]
std=[0.2965,0.1467,0.2864]

train_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
'''
test_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))  
])
'''

train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=train_transforms)

val_pct1=0.8
rand_seed=42

n_val1=int(val_pct1*len(test_dataset))
np.random.seed(rand_seed)
idxs=np.random.permutation(len(test_dataset))

n_val2 = int(n_val1*0.4) # 80% of the test dataset

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=10,shuffle=True)

test_sampler1=SubsetRandomSampler(idxs[:n_val1])
test_loader1 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler1) #Use for model training with 0.8 per target samples

test_sampler2=SubsetRandomSampler(idxs[:n_val2])
test_loader2 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler2) # Use for model training with 0.6 per target samples

test_sampler3=SubsetRandomSampler(idxs[n_val1:])
test_loader3 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler3) # Use for model testing with 0.2 per target samples

class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    #def __init__(self):
    #def __init__(self, input_size1, hidden_size1, num_layers):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        #self.num_layers = num_layers
        #self.hidden_size1 = hidden_size1
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 3)
        #self.conv5 = nn.Conv2d(256, 512, 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3,2)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.3)
        #self.rnn1 = nn.GRU(input_size1, hidden_size1, num_layers, batch_first=True)

    def forward(self, x):
        #h01 = torch.zeros(self.num_layers, 4*x.size(0), self.hidden_size1).to(device)
        x = self.relu(self.bn1(self.conv1(x)))
        #x = self.relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        x = self.dropout(x)
        #print(x.shape)
        x = self.relu(self.bn1(self.conv2(x)))
        #x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1, 256)
        #x, _ = self.rnn1(x, h01)
        #x = self.pool(x)
        #print(x.shape)
        #x = x.reshape(-1,12800)
        #print(x.shape)
        return x


class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.fc1 = nn.Linear(256, 3072)
        self.fc3 = nn.Linear(2048, 7)
        #self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU()
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, latent_size=256, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


'''
class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self):
    #def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(3,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # print(x.shape)
        # x = self.dropout(x)
        # print(x.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        #print(x.shape)
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        #print(x.shape)
        x = x.view(-1,4608)
        #print(x.shape)
        return x


class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.fc1 = nn.Linear(4608, 3072)
        self.fc3 = nn.Linear(2048, 7)
        #self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU()
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x
'''


G = FeatureExtractor().to(device)
C1 = Classifier().to(device)
C2 = Classifier().to(device)

# setup criterion and optimizer
opt_g = torch.optim.Adam(G.parameters(),
                                    lr=0.0001, weight_decay=0.0005)

opt_c1 = torch.optim.Adam(C1.parameters(),
                                     lr=0.0001, weight_decay=0.0005)
opt_c2 = torch.optim.Adam(C2.parameters(),
                                     lr=0.0001, weight_decay=0.0005)


criterion = nn.CrossEntropyLoss()


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))


'''
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))
    load_checkpoint(torch.load("my_checkpoint1.pth"))
    load_checkpoint(torch.load("my_checkpoint2.pth"))
    load_checkpoint(torch.load("my_checkpoint3.pth"))
'''

Dis_train_losses = []
G_train_losses = []
model_train_accuracy = []
patience, trials, best_acc = 5, 0, 1000
training_start_time = time.time()
for epoch in range(100):
    size = 0
    correct = 0
    Dis_running_loss = 0.0
    G_running_loss = 0.0
    C2_running_loss = 0.0
    data_zip = enumerate(zip(train_loader, test_loader2))
    for step, ((images_s, labels_s), (images_t, labels_t)) in data_zip:
        images_s, labels_s, images_t, labels_t = images_s.to(device), labels_s.to(
            device), images_t.to(device), labels_t.to(device)

        # zero gradients for optimizer
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        # compute loss for critic
        feat_s = G(images_s)
        output_s1 = C1(feat_s)
        output_s2 = C2(feat_s)

        loss_s1 = criterion(output_s1, labels_s)
        loss_s2 = criterion(output_s2, labels_s)
        loss_s = loss_s1 + loss_s2
        loss_s.backward()
        opt_g.step()
        opt_c1.step()
        opt_c2.step()

        # zero gradients for optimizer
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        feat_s = G(images_s)
        output_s1 = C1(feat_s)
        output_s2 = C2(feat_s)
        feat_t = G(images_t)
        output_t1 = C1(feat_t)
        output_t2 = C2(feat_t)


        loss_s1 = criterion(output_s1, labels_s)
        loss_s2 = criterion(output_s2, labels_s)
        loss_s = loss_s1 + loss_s2
        loss_dis = discrepancy(output_t1, output_t2)
        loss = loss_s - loss_dis
        loss.backward()
        opt_c1.step()
        opt_c2.step()
        G_running_loss += loss.item()  # * inputs.size(0)

        # zero gradients for optimizer
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        for i in range(4):
            #
            feat_t = G(images_t)
            output_t1 = C1(feat_t)
            output_t2 = C2(feat_t)
            loss_dis = discrepancy(output_t1, output_t2)
            loss_dis.backward()
            opt_g.step()
            Dis_running_loss += loss_dis.item()  # * inputs.size(0)

        c1 = C1(G(images_t))
        c2 = C2(G(images_t))
        c = c1+c2
        _, predicted = torch.max(c, 1)
        correct += (predicted == labels_t).sum().item()
        size += labels_t.size(0)
        step += 1

    G_running_loss = G_running_loss / (len(train_loader))
    Dis_running_loss = Dis_running_loss / (len(test_loader1))
    G_train_losses.append(G_running_loss)
    Dis_train_losses.append(Dis_running_loss)
    tr_accuracy = float(correct / size)
    model_train_accuracy.append(tr_accuracy * 100)
    print(
        f'Epoch:{epoch + 1}, Generator Loss:{G_running_loss:.4f}, Discrepancy Loss:{Dis_running_loss:.4f}, Training Accuracy:{tr_accuracy:.4f}')

    if G_running_loss < best_acc:
        trials = 0
        best_acc = G_running_loss
        # torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
        # logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        epoch_num_Encoder = epoch +1
    else:
        trials += 1
        epoch_num_Encoder = epoch + 1
        if trials >= patience:
            # logging.info(f'Early stopping on epoch {epoch}')
            break

time1 = time.time() - training_start_time
time1 = time1/60
print('Training Time {:.2f}s'.format(time1))
print('Finished Training')
#print('Training Stops at epoch:', epoch_num_Encoder)

with torch.no_grad():
    correct = 0
    size = 0
    raw_correct = 0
    raw_size = 0
    #n_class_correct = [0 for i in range(7)]
    #n_class_samples = [0 for i in range(7)]
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    accuracyT = []
    for idx, (tgt, labels) in enumerate(test_loader3):
        tgt, labels = tgt.to(device), labels.to(device)
        all_labels = torch.cat((all_labels, labels), dim=0)  # For Plotting Purpose in CMT & Hist
        c1 = C1(G(tgt))
        #print(c1)
        #_, predicted1 = torch.max(c1, 1)
        c2 = C2(G(tgt))
        #print(c2)
        c=c1+c2
        #print(c)
        _, predicted = torch.max(c, 1)
        #print(predicted)

        all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT
        correct = (predicted == labels).sum().item()
        size = labels.size(0)
    accuracy = float(correct / size)
print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=classes, zero_division=0))
cm1 = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())  # Making Confusion Matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--pickle1")
    parser.add_argument("-p2", "--pickle2")
    parser.add_argument("-p3", "--pickle3")
    parser.add_argument("-p4", "--pickle4")

    args = parser.parse_args()

    pickle_file1 = args.pickle1
    pickle_file2 = args.pickle2
    pickle_file3 = args.pickle3
    pickle_file4 = args.pickle4


    x1 = []
    for x in range(epoch_num_Encoder):
        x += 1
        x1.append(x)

    plt.figure(dpi=300)
    plt.plot(x1, G_train_losses)
    # plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Generator'], loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    # plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file1)

    plt.figure(dpi=300)
    plt.plot(x1, Dis_train_losses)
    # plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Discrepancy Loss'], loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    # plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file2)

    plt.figure(dpi=300)
    plt.plot(x1,model_train_accuracy)
    #plt.plot(Loss_Val)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['MCD-Model Training Accuracy'],loc='lower right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures/Fig1/Encoder_Accuracy.png')
    plt.savefig(pickle_file3)

    plt.figure(dpi=300)
    plot_confusion_matrix(cm1, classes)
    # plt.savefig('Figures/Fig1/CM_Aligned.png')
    plt.savefig(pickle_file4)

