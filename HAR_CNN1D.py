import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CSIDataset
#from dataset1 import CSIDataset
#from dataset2 import CSIDataset
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 41
learning_rate = 0.00146

input_dim = 936 #468  # Input size with 114 sub-carriers for each antenna pair with 4 such pairs
hidden_dim = 256  # Hidden layer Size
output_dim = 7   # Output size for predicting 7 activities

SEQ_DIM = 32    # Window size
DATA_STEP = 8    # Moving step

BATCH_SIZE = batch_size = 3


classes = ('standing', 'walking', 'get_down', 'sitting',
           'get_up', 'lying', 'no_person')

# Training Data Location
train_dataset = CSIDataset([
        "./dataset2(4,1,3)/bedroom_lviv/4",
    ], SEQ_DIM, DATA_STEP)

#Testing Data Location
test_dataset = CSIDataset([
    "./dataset2(4,1,3)/bedroom_lviv/3",
     ], SEQ_DIM)

val_dataset = CSIDataset([
        "./dataset2(4,1,3)/bedroom_lviv/2",
    ], SEQ_DIM, DATA_STEP)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(32, 16, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(8, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv1d(16, 64, 3)
        self.fc1 = nn.Linear(1760, 120)  #self.fc1 = nn.Linear(1760, 120)  for Mag and Phase separately and 3648 for combined both
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))  # -> n, 6, 14, 14
        # print(x.shape)
        x = self.pool(F.leaky_relu(self.conv2(x)))  # -> n, 16, 5, 5
        # print(x.shape)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, 1760)  # -> n, 400
        # print(x.shape)
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
model = model.double().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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
#print(loss3)
for epoch in range(num_epochs):
    n_correct2 = 0
    n_samples2 = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
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

    #Evalution
    with torch.no_grad():
        n_correct1 = 0
        n_samples1 = 0
        #accuracy = torch.tensor([])
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            '''val_loss2 = torch.tensor([val_loss])
            if (val_loss2 <= val_loss3):
                # print(loss2)
                Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
                val_loss3 = val_loss2'''
            _, predicted = torch.max(outputs, 1)
            #all_preds = torch.cat((all_preds, predicted), dim=0)  # For Plotting Purpose in CMT

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
    if (train_loss2 <= train_loss3 or train_loss2<=0.3):
        # print(loss2)
        Loss_Train = torch.cat((Loss_Train, train_loss2), 0)
        train_loss3 = train_loss2
    else:
        train_loss2=train_loss3
        Loss_Train = torch.cat((Loss_Train, train_loss2), 0)

    val_loss2 = torch.tensor([val_loss])
    if (val_loss2 <= val_loss3 or val_loss2<=0.3):
        # print(loss2)
        Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
        val_loss3 = val_loss2

    else:
        val_loss2=val_loss3
        Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
    #if (i+1) % 50 == 0:
    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Training Accuracy: {acc1T:.4f}, Validation Accuracy: {acc1:.4f}')
#print(accuracy)
#print(val_loss2)
#L_Train = torch.cat((L_Train, train_loss2), 0)
#Loss_Val = torch.cat((Loss_Val, val_loss2), 0)
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
#print(Loss1)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    correct = 0
    n_class_correct = [0 for i in range(7)]
    n_class_samples = [0 for i in range(7)]
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    all_preds1 = torch.tensor([])
    #accuracy = torch.tensor([])
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        all_labels = torch.cat((all_labels, labels), dim=0) #For Plotting Purpose in CMT & Hist

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0) #For Plotting Purpose in CMT

        n_samples += labels.size(0)    # Total Samples
        n_correct += (predicted == labels).sum().item()  # Total Correct Predictions
        #print(n_correct)
        #n_correct1 = torch.tensor([n_correct])
        #acc1 = 100.0 * n_correct / n_samples
        #acc2 = torch.tensor([acc1])
        #accuracy = torch.cat((accuracy, acc2), dim=0)
        #accuracy = torch.cat((accuracy, n_correct1), dim=0)


        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1    # Total Correct Predictions for Individual Class
                label1=torch.tensor([label])
                all_preds1 = torch.cat((all_preds1,label1), dim=0)
            n_class_samples[label] += 1        # Total Samples of each Class

    '''a = torch.zeros([n_class_correct[0]])
    b = torch.ones([n_class_correct[1]])
    c = torch.ones([n_class_correct[2]])
    c=c+1
    d = torch.ones([n_class_correct[3]])
    d=d+2
    e = torch.ones([n_class_correct[4]])
    e=e+3
    f = torch.ones([n_class_correct[5]])
    f=f+4
    g = torch.ones([n_class_correct[6]])
    g=g+5
    all_preds1 = torch.cat((a,b,c,d,e,f,g), dim=0)   #For Plotting Purpose in Hist'''

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(7):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

    cm = confusion_matrix(all_labels,all_preds)  #Making Confusion Matrix

    plt.plot(Loss_Train)
    plt.plot(Loss_Val)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss','Validation Loss'],loc='upper right')
    plt.xlim([0, num_epochs])

    plt.figure()
    plt.plot(accuracyT)
    plt.plot(accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy','Validation Accuracy'], loc='upper left')
    plt.xlim([0, num_epochs])

    df = pd.concat(axis=0, ignore_index=True, objs=[
    pd.DataFrame.from_dict({'value': all_labels, 'name': 'True Labels'}),
    pd.DataFrame.from_dict({'value': all_preds1, 'name': 'Predicted Labels'})
    ])
    fig, ax = plt.subplots()
    sns.histplot(
    data=df, x='value', hue='name', multiple='dodge', shrink=0.8, #stat='density',
    bins=range(0, 8), ax=ax
    )
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.xlabel('Histogram for CSI Magnitude Data')
    plt.ylabel('Counts')

    plt.figure(figsize=(3, 3))
    plot_confusion_matrix(cm, classes)
    plt.show()