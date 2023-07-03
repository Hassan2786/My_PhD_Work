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
import lmmd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epoch_num_Encoder = 0
epoch_num_Aligner = 0
learning_rate = 0.00146

input_size1 = 16 #468  # Input size with 114 sub-carriers for each antenna pair with 4 such pairs
input_size2 = 100
hidden_size1 = 100  # Hidden layer Size
hidden_size2 = 16
output_dim = 50   # Output size for predicting 7 activities
num_layers = 4

BATCH_SIZE = batch_size = 10
load_model=False

classes = ('bending', 'falling', 'lie_down', 'running',
           'sit_down', 'stand_up', 'walking')

train_dataset_path="./Dataset_S1_S3/training/training"
test_dataset_path="./Dataset_S1_S3/validation/validation"

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

n_val2 = int(n_val1*0.2) # 80% of the test dataset

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
        self.fc1 = nn.Linear(256, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 7)
        # self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU()
        self.sf = nn.Softmax(dim=1)
        self.lmmd_loss = lmmd.LMMD_loss(class_num=7)

    def forward(self, s1, t1, s_label):
        #h01 = torch.zeros(self.num_layers, 4*x.size(0), self.hidden_size1).to(device)
        s1 = self.relu(self.bn1(self.conv1(s1)))
        #x = self.relu(self.conv1(x))
        s1 = self.pool(s1)
        # print(x.shape)
        s1 = self.dropout(s1)
        #print(x.shape)
        s1 = self.relu(self.bn1(self.conv2(s1)))
        #x = self.relu(self.conv2(x))
        s1 = self.pool(s1)
        s1 = self.dropout(s1)
        #print(x.shape)
        s1 = self.relu(self.bn2(self.conv3(s1)))
        s1 = self.pool(s1)
        s1 = self.dropout(s1)
        #print(x.shape)
        s1 = self.relu(self.conv4(s1))
        #print(x.shape)
        s1 = s1.view(-1, 256)

        s1 = self.fc1(s1)
        s1 = self.relu(s1)
        s1 = self.fc2(s1)
        s1 = self.relu(s1)
        s_pred = self.fc3(s1)
        s_pred = self.sf(s_pred)

        t1 = self.relu(self.bn1(self.conv1(t1)))
        # x = self.relu(self.conv1(x))
        t1 = self.pool(t1)
        # print(x.shape)
        t1 = self.dropout(t1)
        # print(x.shape)
        t1 = self.relu(self.bn1(self.conv2(t1)))
        # x = self.relu(self.conv2(x))
        t1 = self.pool(t1)
        t1 = self.dropout(t1)
        # print(x.shape)
        t1 = self.relu(self.bn2(self.conv3(t1)))
        t1 = self.pool(t1)
        t1 = self.dropout(t1)
        # print(x.shape)
        t1 = self.relu(self.conv4(t1))
        # print(x.shape)
        t1 = t1.view(-1, 256)
        t1 = self.fc1(t1)
        t1 = self.relu(t1)
        t1 = self.fc2(t1)
        t1 = self.relu(t1)
        t_pred = self.fc3(t1)
        t_pred = self.sf(t_pred)

        loss_lmmd = self.lmmd_loss.get_loss(s1, t1, s_label, t_pred)
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.relu(self.bn1(self.conv2(x)))
        # x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.relu(self.conv4(x))
        # print(x.shape)
        x = x.view(-1, 256)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x

'''


def save_checkpoint(state,filename="my_checkpoint.pth"):
    print("=> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    AE_model.load_state_dict(checkpoint["state_dict"])
    AE_optimiser.load_state_dict(checkpoint['optimizer'])

'''


#F = FeatureExtractor(input_size1,hidden_size1,num_layers).to(device)
F = FeatureExtractor().to(device)
#C = Classifier().to(device)
#D = Discriminator().to(device)


F_opt = torch.optim.Adam(F.parameters(),lr=0.0001)
#C_opt = torch.optim.Adam(C.parameters(),lr=0.0001)
#D_opt = torch.optim.Adam(D.parameters(),lr=0.0001)

#bce = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

'''
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))
    load_checkpoint(torch.load("my_checkpoint1.pth"))
    load_checkpoint(torch.load("my_checkpoint2.pth"))
    load_checkpoint(torch.load("my_checkpoint3.pth"))
'''

max_epoch = 30
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_batches = len(test_loader1)//batch_size
# lamda = 0.01


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.


test_loader_set = iter(test_loader2)

def sample_mnist(step, n_batches):
    global test_loader_set
    if step % n_batches == 0:
        test_loader_set = iter(test_loader2)
    return test_loader_set.next()

Dis_train_losses = []
F_train_losses = []
model_train_accuracy = []
patience, trials, best_acc = 6, 0, 1000

training_start_time = time.time()
for epoch in range(100):

    size = 0
    correct = 0
    #Dis_running_loss = 0.0
    F_running_loss = 0.0
    #C2_running_loss = 0.0
    #data_zip = enumerate(zip(train_loader, test_loader1))
    #for step, ((images_s, labels_s), (images_t, labels_t)) in data_zip:
    for step, (images_s, labels_s) in enumerate(train_loader):  # loop over the dataset multiple times
        images_t, labels_t = sample_mnist(step, n_batches)
        images_s, labels_s, images_t, labels_t = images_s.to(device), labels_s.to(
            device), images_t.to(device), labels_t.to(device)

        # zero gradients for optimizer
        F_opt.zero_grad()

        # compute loss for critic
        label_source_pred, loss_lmmd = F(
            images_s, images_t, labels_s)

        loss_s1 = criterion(label_source_pred, labels_s)
        lambd = get_lambda(epoch,max_epoch)
        loss = loss_s1 + 0.0001*(lambd * loss_lmmd)

        loss.backward()
        F_opt.step()
        F_running_loss += loss.item()  # * inputs.size(0)

        c = F.predict(images_t)
        _, predicted = torch.max(c, 1)
        correct += (predicted == labels_t).sum().item()
        size += labels_t.size(0)
        step += 1

    F_running_loss = F_running_loss / (len(train_loader))
    #Dis_running_loss = Dis_running_loss / (len(test_loader1))
    F_train_losses.append(F_running_loss)
    #Dis_train_losses.append(Dis_running_loss)
    tr_accuracy = float(correct / size)
    model_train_accuracy.append(tr_accuracy * 100)
    print(
        f'Epoch:{epoch + 1}, Feature Extractor Loss:{F_running_loss:.4f}, Training Accuracy:{tr_accuracy:.4f}')

    if F_running_loss < best_acc:
        trials = 0
        best_acc = F_running_loss
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
        c = F.predict(tgt)
        _, predicted = torch.max(c, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT
        correct = (predicted == labels).sum().item()
        size = labels.size(0)
    accuracy = float(correct / size)
print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=classes, zero_division=0))
cm1 = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())  # Making Confusion Matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--pickle1")
    #parser.add_argument("-p2", "--pickle2")
    parser.add_argument("-p3", "--pickle3")
    parser.add_argument("-p4", "--pickle4")

    args = parser.parse_args()

    pickle_file1 = args.pickle1
    #pickle_file2 = args.pickle2
    pickle_file3 = args.pickle3
    pickle_file4 = args.pickle4


    x1 = []
    for x in range(epoch_num_Encoder):
        x += 1
        x1.append(x)

    plt.figure(dpi=300)
    plt.plot(x1, F_train_losses)
    # plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Model Training Loss'], loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    # plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file1)

    '''
    plt.figure(dpi=300)
    plt.plot(x1, C_train_losses)
    # plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Classifier'], loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    # plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file2)
    '''
    plt.figure(dpi=300)
    plt.plot(x1,model_train_accuracy)
    #plt.plot(Loss_Val)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['DSAN-Model Training Accuracy'],loc='lower right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures/Fig1/Encoder_Accuracy.png')
    plt.savefig(pickle_file3)

    plt.figure(dpi=300)
    plot_confusion_matrix(cm1, classes)
    # plt.savefig('Figures/Fig1/CM_Aligned.png')
    plt.savefig(pickle_file4)

