import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from dataset import CSIDataset
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
#from dataset1 import CSIDataset
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 2

learning_rate = 0.00146

input_size = 468 #936 #468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_size = 256
layer_dim = 2
num_classes = 7
sequence_length = 32 #1000 #32
num_layers = 2

SEQ_DIM = 32 #1024 #32
DATA_STEP = 8

BATCH_SIZE = batch_size = 3 #12  #3
#EPOCHS_NUM = 200
LEARNING_RATE = 0.00146

class_weights = torch.Tensor([0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]).double().to(device)
class_weights_inv = 1 / class_weights
# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]

my_transforms=transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize(size=(40,500)),
    #transforms.RandomCrop(size=(32,468)),
    #transforms.ColorJitter(brightness=0.5),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomRotation(degrees=30),#interpolation=PIL.Image.BILINEAR)
    transforms.ToTensor(),
])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
'''train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)'''

classes = ('standing', 'walking', 'get_down', 'sitting',
           'get_up', 'lying', 'no_person')

# Training Data Location
train_dataset = CSIDataset([
        "./Combined_Dataset/R123",
    ], SEQ_DIM, DATA_STEP)

val_pct1=0.15#0.1666383495
val_pct2=0.25444
rand_seed=42

n_val1=int(val_pct1*len(train_dataset))
n_val2=int(val_pct2*len(train_dataset))
np.random.seed(rand_seed)
idxs=np.random.permutation(len(train_dataset))

train_sampler=SubsetRandomSampler(idxs[n_val2:])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,sampler=train_sampler)
val_sampler=SubsetRandomSampler(idxs[:n_val1])
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,sampler=val_sampler)
test_sampler=SubsetRandomSampler(idxs[n_val1:n_val2])
#test_sampler=SubsetRandomSampler(idxs[:n_val2])
test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0,sampler=test_sampler)

'''train_dataset = CSIDataset([
        "./dataset2(4,1,3)/bedroom_lviv/4",
        #"./dataset/bedroom_lviv/4",
        #"./dataset/bedroom_lviv/4",
        # "./dataset/vitalnia_lviv/1/",
        # "./dataset/vitalnia_lviv/2/",
        # "./dataset/vitalnia_lviv/3/",
        #"./dataset/vitalnia_lviv/4/"
    ], SEQ_DIM, DATA_STEP)

#val_dataset = train_dataset
test_dataset = CSIDataset(["./dataset2(4,1,3)/bedroom_lviv/3",
    #"./dataset/bedroom_lviv/3",
#         "./dataset/bedroom_lviv/4",
    #     # "./dataset/vitalnia_lviv/5/"
     ], SEQ_DIM)

val_dataset = CSIDataset([
        "./dataset2(4,1,3)/bedroom_lviv/2",
    ], SEQ_DIM, DATA_STEP)
 #   logging.info("Data is loaded...")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)'''
'''def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))'''


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        h0 = h0.double().to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)
        '''h0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(self.num_layers * 2, batch_size, self.hidden_size).type(torch.DoubleTensor)
        ), requires_grad=True).to(device)

        c0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(self.num_layers * 2, batch_size, self.hidden_size).type(torch.DoubleTensor)
        ), requires_grad=True).to(device)'''
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)
        # or:
        out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # out: (n, 10)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model = model.double().to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_inv)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate ,momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)

# Train the model
n_total_steps = len(train_loader)  # No. of Rows
Loss_Train=torch.tensor([])
Loss_Val=torch.tensor([])
L_Train=torch.tensor([])
L_Val=torch.tensor([])
train_loss3=torch.tensor([3])
val_loss3=torch.tensor([3])
acc3=torch.tensor([0])
accuracy = torch.tensor([])
acc3T=torch.tensor([0])
accuracyT = torch.tensor([])
for epoch in range(num_epochs):
    n_correct2 = 0
    n_samples2 = 0
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        n_samples2 += labels.size(0)  # Total Samples
        n_correct2 += (predicted == labels).sum().item()
        '''train_loss2=torch.tensor([train_loss])
        if(train_loss2<=train_loss3):
            #print(loss2)
            Loss_Train=torch.cat((Loss_Train,train_loss2),0)
            train_loss3 = train_loss2'''
        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    acc1T = n_correct2 / n_samples2
    acc2T = torch.tensor([acc1T])
    if (acc2T >= acc3T):
        accuracyT = torch.cat((accuracyT, acc2T), dim=0)
        acc3T = acc2T
    else:
        acc2T = acc3T
        accuracyT = torch.cat((accuracyT, acc2T), dim=0)
    # Evalution
    with torch.no_grad():
        n_correct1 = 0
        n_samples1 = 0
            # accuracy = torch.tensor([])
        for i, (images, labels) in enumerate(val_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            '''val_loss2 = torch.tensor([val_loss])
                if (val_loss2 <= val_loss3):
                    # print(loss2)
                    Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
                    val_loss3 = val_loss2'''
            _, predicted = torch.max(outputs, 1)
                # all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT

            n_samples1 += labels.size(0)  # Total Samples
            n_correct1 += (predicted == labels).sum().item()
        acc1 = n_correct1 / n_samples1
        acc2 = torch.tensor([acc1])
        if (acc2 >= acc3):
            accuracy = torch.cat((accuracy, acc2), dim=0)
            acc3 = acc2
        else:
            acc2 = acc3
            accuracy = torch.cat((accuracy, acc2), dim=0)

    train_loss2 = torch.tensor([train_loss])
    if (train_loss2 <= train_loss3 or train_loss2 <= 0.3):
            # print(loss2)
        Loss_Train = torch.cat((Loss_Train, train_loss2), 0)
        train_loss3 = train_loss2
    else:
        train_loss2 = train_loss3
        Loss_Train = torch.cat((Loss_Train, train_loss2), 0)

    val_loss2 = torch.tensor([val_loss])
    if (val_loss2 <= val_loss3 or val_loss2 <= 0.3):
            # print(loss2)
        Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
        val_loss3 = val_loss2

    else:
        val_loss2 = val_loss3
        Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
        # if (i+1) % 50 == 0:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Training Accuracy: {acc1T:.4f}, Validation Accuracy: {acc1:.4f}')
    scheduler.step(val_loss)

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    #n_correct1 = 0
    n_samples = 0
    n_class_correct = [0 for i in range(7)]
    n_class_samples = [0 for i in range(7)]
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    all_preds1 = torch.tensor([])
    #accuracy = torch.tensor([])
    #n_class_correct1 = [0 for i in range(7)]
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        all_labels = torch.cat((all_labels, labels), dim=0)  # For Plotting Purpose in CMT & Hist

        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        #n_correct1 = torch.tensor([n_correct])
        #acc1 = 100.0 * n_correct / 4584
        #acc2 = torch.tensor([acc1])
        #accuracy = torch.cat((accuracy, acc2), dim=0)
        #accuracy = torch.cat((accuracy, n_correct1), dim=0)

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            #pred1 = predicted1[i]
            if (label == pred):
                n_class_correct[label] += 1
                label1 = torch.tensor([label])
                all_preds1 = torch.cat((all_preds1, label1), dim=0)
            n_class_samples[label] += 1
            '''if (label == pred1):
                n_class_correct1[label] += 1'''

    #print(n_samples)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    '''acc = 100.0 * n_correct1 / n_samples
    print(f'Accuracy of the network: {acc} %')'''

    for i in range(7):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

    '''for i in range(7):
        acc = 100.0 * n_class_correct1[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')'''

    cm = confusion_matrix(all_labels, all_preds)  # Making Confusion Matrix

    plt.plot(Loss_Train)
    plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlim([0, num_epochs])

    plt.figure()
    plt.plot(accuracyT)
    plt.plot(accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.xlim([0, num_epochs])

    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'value': all_labels, 'name': 'True Labels'}),
        pd.DataFrame.from_dict({'value': all_preds1, 'name': 'Predicted Labels'})
    ])
    fig, ax = plt.subplots()
    sns.histplot(
        data=df, x='value', hue='name', multiple='dodge', shrink=0.8,  # stat='density',
        bins=range(0, 8), ax=ax
    )
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.xlabel('Histogram for CSI Magnitude Data')
    plt.ylabel('Counts')

    plt.figure(figsize=(3, 3))
    plot_confusion_matrix(cm, classes)
    plt.show()