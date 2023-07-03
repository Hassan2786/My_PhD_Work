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

n_val2 = int(n_val1*0.6) # 80% of the test dataset

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=10,shuffle=True)

test_sampler1=SubsetRandomSampler(idxs[:n_val1])
test_loader1 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler1) #Use for model training with 0.8 per target samples

test_sampler2=SubsetRandomSampler(idxs[:n_val2])
test_loader2 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler2) # Use for model training with 0.6 per target samples

test_sampler3=SubsetRandomSampler(idxs[n_val1:])
test_loader3 = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler3) # Use for model testing with 0.2 per target samples

'''
class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self,input_size1,hidden_size1, num_layers,latent_size=4000):
    #def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.unpool = nn.UpsamplingNearest2d(64)
        # self.dropout = nn.Dropout(0.3)
        self.rnn1 = nn.GRU(input_size1, hidden_size1, num_layers, batch_first=True)

        self.fc1 = nn.Linear(64 * 100, latent_size)

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        h01 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        # x = self.dropout(x)
        # print(x.shape)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 64, 16)
        # print(x.shape)
        x, _ = self.rnn1(x, h01)
        # print(x.shape)
        x = x.reshape(-1, 6400)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        return x


class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(2000, 500)
        self.fc1 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU(.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, latent_size=4000, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 250),
            nn.LeakyReLU(0.2),
            nn.Linear(250, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
'''
'''
class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """

    def __init__(self, in_channel=3, hidden_dims=512):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden_dims, 3, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        h = self.conv(x).squeeze()  # (N, hidden_dims)
        return h


class Classifier(nn.Module):
    """
        Classifier
    """

    def __init__(self, input_size=512, num_classes=7):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, input_size=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
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


def save_checkpoint(state,filename="my_checkpoint.pth"):
    print("=> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    AE_model.load_state_dict(checkpoint["state_dict"])
    AE_optimiser.load_state_dict(checkpoint['optimizer'])

'''
'''
class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self):
    #def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 5)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, 3)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        # x = self.dropout(x)
        # print(x.shape)
        #x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1,6400)
        #print(x.shape)
        return x


class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self):
        super(Classifier, self).__init__()
        #self.fc2 = nn.Linear(3072, 2048)
        self.fc1 = nn.Linear(6400, 512)
        self.fc3 = nn.Linear(512, 7)
        #self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU()
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        x = self.fc3(x)
        x = self.sf(x)
        return x


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """

    def __init__(self, latent_size=6400, num_classes=1):
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

#F = FeatureExtractor(input_size1,hidden_size1,num_layers).to(device)
F = FeatureExtractor().to(device)
C = Classifier().to(device)
D = Discriminator().to(device)


F_opt = torch.optim.Adam(F.parameters(),lr=0.0001)
C_opt = torch.optim.Adam(C.parameters(),lr=0.0001)
D_opt = torch.optim.Adam(D.parameters(),lr=0.0001)

bce = nn.BCELoss()
xe = nn.CrossEntropyLoss()

'''
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))
    load_checkpoint(torch.load("my_checkpoint1.pth"))
    load_checkpoint(torch.load("my_checkpoint2.pth"))
    load_checkpoint(torch.load("my_checkpoint3.pth"))
'''

max_epoch = 20
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_batches = len(test_loader1)//batch_size
# lamda = 0.01

D_src = torch.ones(batch_size, 1).to(device) # Discriminator Label to real
D_tgt = torch.zeros(batch_size, 1).to(device) # Discriminator Label to fake
D_labels = torch.cat([D_src, D_tgt], dim=0)

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def get_meu(epoch, max_epoch):
    p = epoch / max_epoch
    return 0.01 / (1+(10*p))**0.75



test_loader_set = iter(test_loader2)

def sample_mnist(step, n_batches):
    global test_loader_set
    if step % n_batches == 0:
        test_loader_set = iter(test_loader2)
    return test_loader_set.next()

fixed_ae_train_losses = []
total_train_losses = []
fixed_ae_test_losses = []
total_test_losses = []
cl_tr_losses = []
cl_tr_accuracy = []
cl_test_losses = []
cl_test_accuracy = []
overall_accuracy = []
model_train_accuracy = []
D_train_losses = []
C_train_losses = []
NUM_EXP = 1
patience, trials, best_acc = 5,0,1000 #1000, 0, 0  #5,0,1000

training_start_time = time.time()
for epoch in range(0, 180):
    size = 0
    correct = 0
    D_running_loss = 0.0
    C_running_loss = 0.0
    Ltot_running_loss = 0.0
    for idx, (src_images, labels) in enumerate(train_loader):  # loop over the dataset multiple times
        tgt_images, tgt_labels = sample_mnist(step, n_batches)
        # Training Discriminator
        src, labels, tgt, t_labels = src_images.to(device), labels.to(device), tgt_images.to(device), tgt_labels.to(device)

        x = torch.cat([src, tgt], dim=0)
        h = F(x)
        y = D(h.detach())

        Ld = bce(y, D_labels)
        D.zero_grad()
        Ld.backward()
        D_opt.step()

        c = C(h[:batch_size])
        y = D(h)
        Lc = xe(c, labels)
        Ld = bce(y, D_labels)
        lamda = 0.1 * get_lambda(epoch, max_epoch)
        Ltot = Lc - lamda * Ld
        c = C(F(src))

        '''
        meu= get_meu(epoch, max_epoch)
        F_opt = torch.optim.Adam(F.parameters(), lr=meu)
        C_opt = torch.optim.Adam(C.parameters(), lr=meu)
        D_opt = torch.optim.Adam(D.parameters(), lr=meu)
        '''

        F.zero_grad()
        C.zero_grad()
        D.zero_grad()

        Ltot.backward()

        C_opt.step()
        F_opt.step()

        D_running_loss += Ld.item()  # * inputs.size(0)
        C_running_loss += Lc.item()  # * inputs.size(0)
        Ltot_running_loss += Ltot.item()  # * inputs.size(0)

        #c1 = C(F(tgt))

        _, predicted = torch.max(c, 1)
        correct += (predicted == labels).sum().item()
        size += labels.size(0)
        step+=1


    D_running_loss = D_running_loss / (len(train_loader))
    C_running_loss = C_running_loss/ (len(train_loader))
    Ltot_running_loss = Ltot_running_loss / (len(train_loader))
    D_train_losses.append(D_running_loss)
    C_train_losses.append(C_running_loss)
    total_train_losses.append(Ltot_running_loss)
    tr_accuracy = float(correct / size)
    model_train_accuracy.append(tr_accuracy * 100)
    print(f'Epoch:{epoch + 1}, Discriminator Loss:{D_running_loss:.4f}, Classifier Loss:{C_running_loss:.4f}, Total Loss:{Ltot_running_loss:.4f}, Training Accuracy:{tr_accuracy:.4f}, lambda: {lamda:.4f}')


    if Ltot_running_loss < best_acc:
        trials = 0
        best_acc = Ltot_running_loss
        # torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
        # logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        epoch_num_Encoder = epoch +1
    else:
        trials += 1
        epoch_num_Encoder = epoch + 1
        if trials >= patience:
            # logging.info(f'Early stopping on epoch {epoch}')
            break
    '''
    if tr_accuracy > best_acc:
        trials = 0
        best_acc = tr_accuracy
        # torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
        # logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        epoch_num_Encoder = epoch + 1
    else:
        trials += 1
        epoch_num_Encoder = epoch + 1
        if trials >= patience:
            # logging.info(f'Early stopping on epoch {epoch}')
            break
    '''

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
        c = C(F(tgt))
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
    plt.plot(x1, D_train_losses)
    # plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss of Discriminator'], loc='upper right')
    plt.xlim([1, epoch_num_Encoder])
    # plt.savefig('Figures_M_1/T1/Encoder_loss.png')
    plt.savefig(pickle_file1)

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

    plt.figure(dpi=300)
    plt.plot(x1,model_train_accuracy)
    #plt.plot(Loss_Val)
    plt.title('Model Accuracy')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['DAN-Model Training Accuracy'],loc='lower right')
    plt.xlim([1, epoch_num_Encoder])
    #plt.savefig('Figures/Fig1/Encoder_Accuracy.png')
    plt.savefig(pickle_file3)

    plt.figure(dpi=300)
    plot_confusion_matrix(cm1, classes)
    # plt.savefig('Figures/Fig1/CM_Aligned.png')
    plt.savefig(pickle_file4)

